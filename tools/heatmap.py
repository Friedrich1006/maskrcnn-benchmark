import argparse
import os
import pickle

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.projection import projection
from maskrcnn_benchmark.data.transforms import transforms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.img import ImageProc


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument(
    "--config-file",
    default="configs/vod/stage1.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cfg.OUTPUT_DIR = 'heatmap'
cfg.freeze()

print('Building model...')
model = build_detection_model(cfg)
backbone = model.backbone
model = model.rpn.rnn
model.load_state_dict(torch.load('stage1_rnn_mse/model_final.pth')['model'])
device = torch.device(cfg.MODEL.DEVICE)
backbone.to(device)
model.to(device)

print('Building dataset...')
root = 'datasets/ILSVRC2015'
task_set = 'VID'
split = 'val'
img_index = 'VID_val_all'
anno_path = os.path.join(root, 'Annotations', task_set, split, '%s.xml')
img_path = os.path.join(root, 'Data', task_set, split, '%s.JPEG')
imgset_path = os.path.join(root, 'ImageSets', '%s.txt')
cache_path = os.path.join(root, '__cache__')
cache_file = os.path.join(cache_path, img_index + '_anno.pkl')
transform = transforms.Compose(
    [
        transforms.Resize(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN),
        transforms.ToTensor(),
    ]
)

with open(imgset_path % img_index) as f:
    file_idx = [x.strip('\n') for x in f.readlines()]

with open(cache_file, 'rb') as fid:
    annos = pickle.load(fid)

output_dir = cfg.OUTPUT_DIR
proc = ImageProc()

start = 161168
with torch.no_grad():
    for idx in range(100):
        file_name = file_idx[start + idx]
        img = Image.open(img_path % file_name).convert('RGB')
        anno = annos[start + idx]
        height, width = anno['im_info']
        target = BoxList(anno['boxes'].reshape(-1, 4), (width, height), mode='xyxy')
        target.add_field('labels', anno['labels'])
        target.add_field('scores', torch.ones(len(target)))
        target.add_field('objectness', torch.ones(len(target)))
        target = target.clip_to_image(remove_empty=True)
        img, target = transform(img, target)
        if idx == 0:
            images = to_image_list(img.unsqueeze(0).to(device))
            features = backbone(images.tensors)
            feature_h = features[0].size(2)
            feature_w = features[0].size(3)
            heatmap = torch.zeros(1, 1, feature_h, feature_w, device=device)
            proj = projection(target, (feature_h, feature_w), 0.5).to(device)
            last_state = model(proj)

        else:
            heatmap = last_state[-1][0]
            proj = projection(target, (feature_h, feature_w), 0.5).to(device)
            last_state = model(proj, last_state)

        img = (img[[2, 1, 0]].permute(1, 2, 0) * 255).numpy().astype('uint8')
        img1 = img.copy()
        img1 = proc.overlay_boxes(img1, target)
        img2 = img.copy()
        img2 = proc.overlay_heatmap(img2, proj)
        img3 = img.copy()
        img3 = proc.overlay_heatmap(img3, heatmap)
        cv2.imwrite(os.path.join(output_dir, '1_{:0>8d}.png'.format(start + idx)), img1)
        cv2.imwrite(os.path.join(output_dir, '2_{:0>8d}.png'.format(start + idx)), img2)
        cv2.imwrite(os.path.join(output_dir, '3_{:0>8d}.png'.format(start + idx)), img3)
