# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.convlstm import ConvLSTM, ConvLSTMCell
from maskrcnn_benchmark.modeling.projection import projection


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, videos=None, frames=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


class ROIBoxHeadVideo(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHeadVideo, self).__init__()

        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if cfg.MODEL.ROI_BOX_HEAD.RNN.COMBINATION == 'cat':
            in_channels += self.num_classes
        elif cfg.MODEL.ROI_BOX_HEAD.RNN.COMBINATION == 'attention_norm':
            self.topk = cfg.MODEL.ROI_BOX_HEAD.RNN.TOPK

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.rnn = ConvLSTM(cfg.MODEL.ROI_BOX_HEAD.RNN.INPUT_DIM,
                            cfg.MODEL.ROI_BOX_HEAD.RNN.HIDDEN_DIM,
                            cfg.MODEL.ROI_BOX_HEAD.RNN.KERNEL_SIZE,
                            cfg.MODEL.ROI_BOX_HEAD.RNN.NUM_LAYERS,
                            cfg.MODEL.ROI_BOX_HEAD.RNN.BIAS,
                            cfg.MODEL.ROI_BOX_HEAD.RNN.PRETRAIN)
        self.project_th = cfg.MODEL.ROI_BOX_HEAD.RNN.PROJECT_TH
        self.combination = getattr(self, cfg.MODEL.ROI_BOX_HEAD.RNN.COMBINATION)
        self.last_state = None
        self.video = ''

    def forward(self, features, proposals, targets=None, videos=None, frames=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
            videos (list[str]): video names of images (optional)
            frames (list[int]): frame indices of images in videos (optional)

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        feature_h = features[0].size(2)
        feature_w = features[0].size(3)
        device = features[0].device
        x = []
        results = []
        losses = {}
        for idx in range(features[0].size(0)):
            if self.video == videos[idx]:
                heatmap = self.last_state[-1][0]
            else:
                heatmap = torch.zeros(self.num_classes, 1, feature_h, feature_w,
                                      device=device)
            features_i = features[0][idx].unsqueeze(0)
            comb = self.combination(features_i, heatmap)
            proposals_i = [proposals[idx]]
            targets_i = [targets[idx]]

            if self.training:
                # Faster R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    proposals_i = self.loss_evaluator.subsample(proposals_i, targets_i)

            if self.combination == self.attention_norm:
                heatsum = heatmap.view(self.num_classes, -1).sum(dim=1)
                _, top_c = heatsum.topk(self.topk)
                comb = comb[top_c]

                x_i = []
                class_logits_i = []
                box_regression_i = []

                for c in range(self.topk):
                    x_i_c = self.feature_extractor([comb[c: c + 1]], proposals_i)
                    class_logits_i_c, box_regression_i_c = self.predictor(x_i_c)
                    x_i.append(x_i_c)
                    class_logits_i.append(class_logits_i_c)
                    box_regression_i.append(box_regression_i_c)

                x_i = sum(x_i) / self.topk
                class_logits_i = sum(class_logits_i) / self.topk
                box_regression_i = sum(box_regression_i) / self.topk

            else:
                # extract features that will be fed to the final classifier. The
                # feature_extractor generally corresponds to the pooler + heads
                x_i = self.feature_extractor(comb, proposals_i)
                # final classifier that converts the features into predictions
                class_logits_i, box_regression_i = self.predictor(x_i)

            proposals_i = self.post_processor((class_logits_i, box_regression_i), proposals_i)

            loss_classifier_i, loss_box_reg_i = self.loss_evaluator(
                [class_logits_i], [box_regression_i]
            )
            losses_i = {
                "loss_classifier": loss_classifier_i,
                "loss_box_reg": loss_box_reg_i,
            }

            proj = projection(proposals_i[0], (feature_h, feature_w),
                              self.project_th, self.num_classes)
            if self.video == videos[idx]:
                self.last_state = self.rnn(proj, self.last_state)
            else:
                self.last_state = self.rnn(proj)
                self.video = videos[idx]
            x.append(x_i)
            results.extend(proposals_i)
            for k, v in losses_i.items():
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

        x = torch.cat(x, dim=0)
        return x, results, losses

    def cat(self, feature, heatmap):
        return torch.cat((feature, heatmap), dim=1)

    def attention_norm(self, feature, heatmap):
        from math import sqrt
        b, c, h, w = heatmap.shape
        att = F.softmax(heatmap.view(-1) / sqrt(b * c * h * w), dim=0).view(b, c, h, w)
        feature_att = feature * att
        return feature_att


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    if cfg.MODEL.ROI_BOX_HEAD.VIDEO_ON:
        return ROIBoxHeadVideo(cfg, in_channels)
    return ROIBoxHead(cfg, in_channels)
