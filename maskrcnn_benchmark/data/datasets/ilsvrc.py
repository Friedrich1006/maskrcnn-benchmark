import numpy as np
import os
import pickle
import random
import sys

import torch
import torch.utils.data
from torchvision.transforms import functional as F

from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class ILSVRCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = ['__background__',  # always index 0
                    'airplane', 'antelope', 'bear', 'bicycle',
                    'bird', 'bus', 'car', 'cattle',
                    'dog', 'domestic_cat', 'elephant', 'fox',
                    'giant_panda', 'hamster', 'horse', 'lion',
                    'lizard', 'monkey', 'motorcycle', 'rabbit',
                    'red_panda', 'sheep', 'snake', 'squirrel',
                    'tiger', 'train', 'turtle', 'watercraft',
                    'whale', 'zebra']
    CLASSES_IDX = ['__background__',  # always index 0
                   'n02691156', 'n02419796', 'n02131653', 'n02834778',
                   'n01503061', 'n02924116', 'n02958343', 'n02402425',
                   'n02084071', 'n02121808', 'n02503517', 'n02118333',
                   'n02510455', 'n02342885', 'n02374451', 'n02129165',
                   'n01674464', 'n02484322', 'n03790512', 'n02324045',
                   'n02509815', 'n02411705', 'n01726692', 'n02355227',
                   'n02129604', 'n04468005', 'n01662784', 'n04530566',
                   'n02062744', 'n02391049']

    def __init__(self, root, task_set, split, img_index, transforms=None,
                 use_anno_cache=True):
        self.root = root
        self.task_set = task_set
        self.split = split
        self.img_index = img_index
        self.transforms = transforms

        self.anno_path = os.path.join(root, 'Annotations', task_set, split,
                                       '%s.xml')
        self.img_path = os.path.join(root, 'Data', task_set, split, '%s.JPEG')
        self.imgset_path = os.path.join(root, 'ImageSets', '%s.txt')

        with open(self.imgset_path % img_index) as f:
            self.file_idx = [x.strip('\n') for x in f.readlines()]
        self.frame_idx = [int(x.split('/')[-1]) for x in self.file_idx]

        self.classes_idx_to_lbl = dict(zip(self.CLASSES_IDX,
                                           range(len(self.CLASSES_IDX))))

        self.use_anno_cache = use_anno_cache
        if self.use_anno_cache:
            self.cache_path = os.path.join(root, '__cache__')
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            cache_file = os.path.join(self.cache_path, self.img_index + '_anno.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    self.annos = pickle.load(fid)
                print('{}\'s annotation loaded from {}'.format(self.img_index, cache_file))
            else:
                self.annos = self.preload_annos()
                with open(cache_file, 'wb') as fid:
                    pickle.dump(self.annos, fid)
                print('{}\'s annotation saved to {}'.format(self.img_index, cache_file))

    def __getitem__(self, idx):
        file_name = self.file_idx[idx]
        img = Image.open(self.img_path % file_name).convert('RGB')

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        video = file_name.rsplit('/', 1)[0]
        frame = self.frame_idx[idx]

        return img, target, (idx, file_name, video, frame)

    def __len__(self):
        return len(self.file_idx)

    def get_groundtruth(self, idx):
        if self.use_anno_cache:
            anno = self.annos[idx]
        else:
            file_name = self.file_idx[idx]
            tree = ET.parse(self.anno_path % file_name).getroot()
            anno = self.preprocess_annotation(tree)

        height, width = anno['im_info']
        target = BoxList(anno['boxes'].reshape(-1, 4), (width, height), mode='xyxy')
        target.add_field('labels', anno['labels'])

        return target

    def preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find('size')
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))

        objs = target.findall('object')
        for obj in objs:
            if not obj.find('name').text in self.classes_idx_to_lbl:
                continue

            bbox = obj.find('bndbox')
            box = [
                np.maximum(float(bbox.find('xmin').text), 0),
                np.maximum(float(bbox.find('ymin').text), 0),
                np.minimum(float(bbox.find('xmax').text), im_info[1] - 1),
                np.minimum(float(bbox.find('ymax').text), im_info[0] - 1)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_idx_to_lbl[obj.find('name').text.lower().strip()])

        res = {
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(gt_classes),
            'im_info': im_info,
        }
        return res

    def preload_annos(self):
        annos = []
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print('Processed {} images'.format(idx))

            filename = self.file_idx[idx]

            tree = ET.parse(self.anno_path % filename).getroot()
            anno = self.preprocess_annotation(tree)
            annos.append(anno)
        print('Processed {} images'.format(len(self)))
        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]['im_info']
        return {'height': im_info[0], 'width': im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return ILSVRCDataset.CLASSES_NAME[class_id]

    def shuffle_videos(self):
        n = len(self.file_idx)
        start_idx = []
        for idx in range(n):
            if self.frame_idx[idx] == 0:
                start_idx.append(idx)
        random.shuffle(start_idx)
        new_file_idx = []
        new_annos = []
        new_frame_idx = []
        for start in start_idx:
            for idx in range(start, n):
                if self.frame_idx[idx] == 0 and idx != start:
                    break
                new_file_idx.append(self.file_idx[idx])
                new_annos.append(self.annos[idx])
                new_frame_idx.append(self.frame_idx[idx])
        self.file_idx = new_file_idx
        self.annos = new_annos
        self.frame_idx = new_frame_idx
