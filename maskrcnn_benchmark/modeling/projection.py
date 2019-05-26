import numpy as np
import torch
from math import floor, ceil

def projection(boxes, feature_size, threshold, num_classes=1):
    device = boxes.bbox.device

    y_ratio = float(feature_size[0]) / float(boxes.size[1])
    x_ratio = float(feature_size[1]) / float(boxes.size[0])
    ratio = torch.tensor([x_ratio, y_ratio, x_ratio, y_ratio], device=device)
    resized_boxes = boxes.bbox * ratio
    resized_boxes = resized_boxes.detach().cpu()

    if num_classes == 1:
        objectness = boxes.get_field('objectness').detach().cpu().numpy()
    else:
        objectness = boxes.get_field('scores').detach().cpu().numpy()
        labels = boxes.get_field('labels').detach().cpu().numpy()

    proj = torch.zeros(num_classes, feature_size[0], feature_size[1], device=device)

    for i in range(resized_boxes.size(0)):
        o = objectness[i]
        if o > threshold:
            box = resized_boxes[i].numpy()
            x_min = max(box[0], 0)
            y_min = max(box[1], 0)
            x_max = min(box[2], feature_size[1] - 1e-6)
            y_max = min(box[3], feature_size[0] - 1e-6)

            x1 = floor(x_min)
            x2 = ceil(x_min)
            x3 = floor(x_max)
            x4 = ceil(x_max)
            xp = x2 - x_min
            xq = x_max - x3
            y1 = floor(y_min)
            y2 = ceil(y_min)
            y3 = floor(y_max)
            y4 = ceil(y_max)
            yp = y2 - y_min
            yq = y_max - y3

            if num_classes == 1:
                l = 0
            else:
                l = labels[i]

            proj[l, y2: y3, x2: x3] += o
            proj[l, y1: y2, x2: x3] += o * yp
            proj[l, y3: y4, x2: x3] += o * yq
            proj[l, y2: y3, x1: x2] += o * xp
            proj[l, y2: y3, x3: x4] += o * xq
            proj[l, y1: y2, x1: x2] += o * yp * xp
            proj[l, y1: y2, x3: x4] += o * yp * xq
            proj[l, y3: y4, x1: x2] += o * yq * xp
            proj[l, y3: y4, x3: x4] += o * yq * xq
    return proj.view(num_classes, 1, feature_size[0], feature_size[1]).detach()
