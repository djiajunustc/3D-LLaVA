import torch
import torch.nn as nn
from .builder import build_criteria
from .structure import InstanceData_
from .builder import LOSSES


@LOSSES.register_module()
class PerceptionCriterion(nn.Module):

    def __init__(self, ref_criterion=None):
        super().__init__()

        self.ref_criterion = build_criteria(cfg=ref_criterion)
        
    def forward(self, pred_dict, query_masks, gt_masks, gt_labels):
        """Calculate loss.
        """

        gts = []
        for i in range(len(gt_masks)):
            inst_gt = InstanceData_()
            inst_gt.sp_masks = gt_masks[i]
            inst_gt.labels_3d = gt_labels[i]
            if query_masks is not None:
                inst_gt.query_masks = query_masks[i]
            gts.append(inst_gt)
        
        seg_loss = self.ref_criterion(pred_dict, gts)

        return seg_loss
