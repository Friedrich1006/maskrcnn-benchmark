from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer_tb import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.structures.image_list import to_image_list


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument(
    "--config-file",
    default="configs/vod/stage3.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)

parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)

parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

cfg.MODEL.RPN.RNN.COMBINATION = 'attention_norm'
cfg.OUTPUT_DIR = 'stage3_attention_norm'
cfg.DATASETS.USE_ANNO_CACHE = False

cfg.freeze()

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

optimizer = make_optimizer(cfg, model)
scheduler = make_lr_scheduler(cfg, optimizer)

if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
    )

arguments = {}
arguments["iteration"] = 0

output_dir = cfg.OUTPUT_DIR

save_to_disk = get_rank() == 0
checkpointer = DetectronCheckpointer(
    cfg, model, optimizer, scheduler, output_dir, save_to_disk
)

extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
arguments.update(extra_checkpoint_data)

data_loader = make_data_loader(
    cfg,
    is_train=True,
    is_distributed=args.distributed,
    start_iter=arguments["iteration"],
)

checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

model.train()
itr = iter(data_loader)
images, targets, others = itr.next()
images = images.to(device)
targets = [target.to(device) for target in targets]
videos = [o[2] for o in others]
frames = [o[3] for o in others]

backbone = model.backbone
rpn = model.rpn
roi_heads = model.roi_heads
box = roi_heads.box
rnn = box.rnn

images = to_image_list(images)
features = backbone(images.tensors)