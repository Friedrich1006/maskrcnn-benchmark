from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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

import datetime
import logging
import time

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.projection import projection
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument(
    "--config-file",
    default="configs/vod/stage1.yaml",
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

cfg.DATASETS.TRAIN = ('VID_train_seg16', )
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.OUTPUT_DIR = 'stage1_rnn_mse'

cfg.freeze()

output_dir = cfg.OUTPUT_DIR

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

model = build_detection_model(cfg)
backbone = model.backbone
model = model.rpn.rnn
device = torch.device(cfg.MODEL.DEVICE)
backbone.to(device)
model.to(device)

optimizer = make_optimizer(cfg, model)
scheduler = make_lr_scheduler(cfg, optimizer)

arguments = {}
arguments["iteration"] = 0

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

if __name__ == '__main__':
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    writer = SummaryWriter(os.path.join(checkpointer.save_dir, 'logs'))
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    last_state = None
    video = ''
    feature_h = 0
    feature_w = 0
    loss = torch.zeros(1, device=device)
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, others) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        targets = [target.to(device) for target in targets]
        videos = [o[2] for o in others]
        frames = [o[3] for o in others]

        for i in range(len(targets)):
            boxes = targets[i]
            boxes.add_field('objectness', torch.ones(len(targets[i]), device=device))
            if video == videos[i]:
                heatmap = last_state[-1][0]
                proj = projection(boxes, (feature_h, feature_w))
                loss += F.mse_loss(heatmap, proj)
                last_state = model(proj, last_state)

            else:
                meters.update(loss=loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                images = images.to(device)
                images = to_image_list(images)
                with torch.no_grad():
                    features = backbone(images.tensors)
                    feature_h = features[0].size(2)
                    feature_w = features[0].size(3)

                heatmap = torch.zeros(1, 1, feature_h, feature_w, device=device)
                proj = projection(boxes, (feature_h, feature_w))
                loss = torch.zeros(1, device=device)
                last_state = model(proj)
                video = videos[i]

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            for name, meter in meters.meters.items():
                writer.add_scalar('train/{}'.format(name), meter.median, iteration)
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
