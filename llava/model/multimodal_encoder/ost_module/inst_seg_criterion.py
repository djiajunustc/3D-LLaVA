import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

from .structure import InstanceData_
from .builder import LOSSES, TASK_UTILS


def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))

    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def compute_iou(pred_masks, true_mask):
    """
    计算预测的masks与真实mask之间的IoU

    参数:
    - pred_masks: 预测的masks, 维度 (N, M)
    - true_mask: 真实的mask, 维度 (1, M)

    返回:
    - ious: 每个预测mask与真实mask之间的IoU, 维度 (N,)
    """
    pred_masks = pred_masks > 0.5
    
    # 将真实mask扩展到与预测mask相同的形状
    true_mask = true_mask.expand_as(pred_masks)

    # 计算交集 (intersection)
    intersection = (pred_masks & true_mask).float().sum(dim=1, keepdim=True)

    # 计算并集 (union)
    union = (pred_masks | true_mask).float().sum(dim=1, keepdim=True)

    # 计算IoU
    iou = intersection / (union + 1e-6)

    return iou


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


@LOSSES.register_module()
class InstanceCriterion(nn.Module):
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(self, matcher, loss_weight, non_object_weight, 
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss=False, aux_top_k=-1):
        super().__init__()
        self.matcher = TASK_UTILS.build(matcher)
        # class_weight = [1] * num_classes + [non_object_weight]
        # self.class_weight = class_weight
        self.non_object_weight = non_object_weight
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.aux_top_k = aux_top_k

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            class_weight = [1] * n_classes + [self.non_object_weight]
            cls_losses.append(F.cross_entropy(
                cls_pred.float(), cls_target, cls_pred.new_tensor(class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask.float(), tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask.float(), tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData_
    def forward(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']
    
        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            class_weight = [1] * n_classes + [self.non_object_weight]
            cls_losses.append(F.cross_entropy(
                cls_pred.float(), cls_target, cls_pred.new_tensor(class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask.float(), tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask.float(), tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                if self.aux_top_k < 0 or len(pred['aux_outputs']) - i <= self.aux_top_k:
                    loss += self.get_layer_loss(aux_outputs, insts, indices)

        return loss


@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.float().softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks.detach().float(), gt_instances.masks.detach().float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks.detach().float(), gt_instances.masks.detach().float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskIoUCost:
    """IoU cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = 1 - compute_iou(pred_instances.masks, gt_instances.masks)
        return cost * self.weight


@TASK_UTILS.register_module()
class OptMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """
    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)
        
        # values = torch.topk(
        #     cost_value, self.topk + 1, dim=0, sorted=True,
        #     largest=False).values[-1:, :]
        # ids = torch.argwhere(cost_value < values)

        # import pdb
        # pdb.set_trace()
        pair_wise_ious = compute_iou(pred_instances.masks, gt_instances.masks)
        
        ids, matched_qidx = self.dynamic_k_matching(cost_value, pair_wise_ious, n_gts)

        return ids[0], ids[1]
        

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost) # [300,num_gt] 
        ious_in_boxes_matrix = pair_wise_ious
        n_query = len(ious_in_boxes_matrix)
        n_candidate_k = min(n_query, 10)
        
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)

        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:,gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:,gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)
        
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1) 
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 

        while (matching_matrix.sum(0)==0).any(): 
            num_zero_gt = (matching_matrix.sum(0)==0).sum()
            matched_query_id = matching_matrix.sum(1)>0
            cost[matched_query_id] += 100000.0 
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:,gt_idx])
                matching_matrix[:,gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0: 
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
                matching_matrix[anchor_matching_gt > 1] *= 0 
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1 

        assert not (matching_matrix.sum(0)==0).any() 
        # 
        selected_query = matching_matrix.sum(1)>0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix==0] = cost[matching_matrix==0] + float('inf')
        matched_query_id = torch.min(cost,dim=0)[1]

        # in this version, the selected_query is [300,] with true/false
        # we convert it to value indices
        matched_anchor_inds = torch.arange(len(matching_matrix)).to(gt_indices)
        selected_query = matched_anchor_inds[selected_query]     # [num_pos,]

        return (selected_query,gt_indices) , matched_query_id


@TASK_UTILS.register_module()
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """
    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))

        cost_value = torch.stack(cost_values).sum(dim=0)

        max_value = 1e6  
        cost_value = torch.clamp(cost_value, min=-1*max_value, max=max_value)

        cost_value = torch.where(torch.isnan(cost_value), torch.full_like(cost_value, max_value), cost_value)
        cost_value = torch.where(torch.isinf(cost_value), torch.full_like(cost_value, max_value), cost_value)

        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())

        device = labels.device
        query_ids = torch.tensor(query_ids).to(device)
        object_ids = torch.tensor(object_ids).to(device)
        return query_ids, object_ids


@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        
        if hasattr(gt_instances, 'query_masks'):
            cost_value = torch.where(
                gt_instances.query_masks.T, cost_value, self.inf)

            values = torch.topk(
                cost_value, self.topk + 1, dim=0, sorted=True,
                largest=False).values[-1:, :]

            ids = torch.argwhere(cost_value < values)
            return ids[:, 0], ids[:, 1]
        else:
            cost_value = cost_value.cpu().numpy()
            cost_value = np.nan_to_num(cost_value, nan=1e5, posinf=1e5, neginf=1e5)
            query_ids, object_ids = linear_sum_assignment(cost_value)
            device = labels.device
            query_ids = torch.tensor(query_ids).to(device)
            object_ids = torch.tensor(object_ids).to(device)
            return query_ids, object_ids


@LOSSES.register_module()
class OneDataCriterion:
    """Loss module for SPFormer.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
    """

    def __init__(self, matcher, loss_weight, non_object_weight, 
                 num_classes_1dataset, num_classes_2dataset,
                 fix_dice_loss_weight, iter_matcher):
        self.matcher = TASK_UTILS.build(matcher)
        self.num_classes_1dataset = num_classes_1dataset
        self.num_classes_2dataset = num_classes_2dataset
        self.class_weight_1dataset = [1] * num_classes_1dataset + [non_object_weight]
        self.class_weight_2dataset = [1] * num_classes_2dataset + [non_object_weight]
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            if cls_pred.shape[1] == self.num_classes_1dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_1dataset)))
            elif cls_pred.shape[1] == self.num_classes_2dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_2dataset)))
            else:
                raise RuntimeError(
                    f'Invalid classes number {cls_pred.shape[1]}.')

        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
        mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData
    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_gts, n_classes + 1);
                List `scores` of len batch_size each of shape (n_gts, 1);
                List `masks` of len batch_size each of shape (n_gts, n_points).
                Dict `aux_preds` with list of cls_preds, scores, and masks.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]            
            if cls_pred.shape[1] == self.num_classes_1dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_1dataset)))
            elif cls_pred.shape[1] == self.num_classes_2dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_2dataset)))
            else:
                raise RuntimeError(
                    f'Invalid classes number {cls_pred.shape[1]}.')
        
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
        mask_dice_loss = torch.stack(mask_dice_losses).sum()

        if self.fix_dice_loss_weight:
            mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}
