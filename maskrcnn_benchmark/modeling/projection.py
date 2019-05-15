import torch
from math import floor, ceil

def projection(boxes, feature_size):
    device = boxes.bbox.device

    x_ratio = float(feature_size[0]) / float(boxes.size[1])
    y_ratio = float(feature_size[1]) / float(boxes.size[0])
    ratio = torch.tensor([x_ratio, y_ratio, x_ratio, y_ratio], device=device)
    resized_boxes = boxes.bbox * ratio
    resized_boxes = resized_boxes.cpu()

    objectness = boxes.get_field('objectness').cpu().numpy()

    proj = torch.zeros(feature_size[0], feature_size[1], device=device)

    for i in range(resized_boxes.size(0)):
        box = resized_boxes[i].numpy()
        x_min = max(box[0], 0)
        y_min = max(box[1], 0)
        x_max = min(box[2], feature_size[0] - 1e-6)
        y_max = min(box[3], feature_size[1] - 1e-6)

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

        o = objectness[i]
        proj[y2: y3, x2: x3] += o
        proj[y1: y2, x2: x3] += o * yp
        proj[y3: y4, x2: x3] += o * yq
        proj[y2: y3, x1: x2] += o * xp
        proj[y2: y3, x3: x4] += o * xq
        proj[y1: y2, x1: x2] += o * yp * xp
        proj[y1: y2, x3: x4] += o * yp * xq
        proj[y3: y4, x1: x2] += o * yq * xp
        proj[y3: y4, x3: x4] += o * yq * xq
    return proj.view(1, 1, feature_size[0], feature_size[1])
