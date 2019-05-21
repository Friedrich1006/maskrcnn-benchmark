# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from maskrcnn_benchmark.modeling.convlstm import ConvLSTM, ConvLSTMCell
from maskrcnn_benchmark.modeling.projection import projection


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, videos=None, frames=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


class RPNVideoModule(torch.nn.Module):
    """
    Module for RPN computation in videos.
    """

    def __init__(self, cfg, in_channels):
        super(RPNVideoModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        if cfg.MODEL.RPN.RNN.COMBINATION == 'cat':
            in_channels += 1
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

        self.rnn = ConvLSTM(cfg.MODEL.RPN.RNN.INPUT_DIM,
                            cfg.MODEL.RPN.RNN.HIDDEN_DIM,
                            cfg.MODEL.RPN.RNN.KERNEL_SIZE,
                            cfg.MODEL.RPN.RNN.NUM_LAYERS,
                            cfg.MODEL.RPN.RNN.BIAS,
                            cfg.MODEL.RPN.RNN.PRETRAIN)
        self.project_th = cfg.MODEL.RPN.RNN.PROJECT_TH
        self.combination = getattr(self, cfg.MODEL.RPN.RNN.COMBINATION)
        self.last_state = None
        self.video = ''

    def forward(self, images, features, targets=None, videos=None, frames=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            videos (list[str]): video names of images (optional)
            frames (list[int]): frame indices of images in videos (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        anchors = self.anchor_generator(images, features)
        feature_h = features[0].size(2)
        feature_w = features[0].size(3)
        device = features[0].device
        boxes = []
        losses = {}
        for idx in range(features[0].size(0)):
            if self.video == videos[idx]:
                heatmap = self.last_state[-1][0]
            else:
                heatmap = torch.zeros(1, 1, feature_h, feature_w, device=device)
            features_i = features[0][idx].unsqueeze(0)
            comb = [self.combination(features_i, heatmap)]
            anchors_i = [anchors[idx]]
            targets_i = [targets[idx]]
            objectness_i, rpn_box_regression_i = self.head(comb)
            if self.training:
                boxes_i, losses_i = self._forward_train(anchors_i, objectness_i,
                                                        rpn_box_regression_i, targets_i)
                proj = projection(boxes_i[0], (feature_h, feature_w), self.project_th)
                if self.video == videos[idx]:
                    self.last_state = self.rnn(proj, self.last_state)
                else:
                    self.last_state = self.rnn(proj)
                    self.video = videos[idx]
                boxes.extend(boxes_i)
                for k, v in losses_i.items():
                    if k in losses:
                        losses[k] += v
                    else:
                        losses[k] = v
            else:
                boxes_i, _ = self._forward_test(anchors, objectness_i, rpn_box_regression_i)
                proj = projection(boxes_i[0], (feature_h, feature_w), self.project_th)
                if self.video == videos[idx]:
                    self.last_state = self.rnn(proj, self.last_state)
                else:
                    self.last_state = self.rnn(proj)
                    self.video = videos[idx]
                boxes.extend(boxes_i)
        return boxes, losses

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}

    def attention(self, feature, heatmap):
        b, c, h, w = heatmap.shape
        att = F.softmax(heatmap.view(b, c, -1), dim=2).view(b, c, h, w)
        feature_att = feature * att
        return feature_att

    def attention_norm(self, feature, heatmap):
        from math import sqrt
        b, c, h, w = heatmap.shape
        att = F.softmax(heatmap.view(-1) / sqrt(b * c * h * w), dim=0).view(b, c, h, w)
        feature_att = feature * att
        return feature_att

    def cat(self, feature, heatmap):
        return torch.cat((feature, heatmap), dim=1)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)
    if cfg.MODEL.RPN.VIDEO_ON:
        return RPNVideoModule(cfg, in_channels)
    return RPNModule(cfg, in_channels)
