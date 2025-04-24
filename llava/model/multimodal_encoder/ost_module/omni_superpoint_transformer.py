import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

import numpy as np

from .builder import MODELS, build_model
from .mask_matrix_nms import mask_matrix_nms

from torch_geometric.utils import scatter
from pointgroup_ops import voxelization
from ..custom_spconv_module.spconv_layers import SubMConv3d
from ..custom_spconv_module.batchnorm import BatchNorm1d
from ..spconv_unet import SpConvUNet
from pointops import farthest_point_sampling


@MODELS.register_module()
class OmniSuperPointTransformer(nn.Module):
    def __init__(self, 
                 backbone=None, 
                 decoder=None, 
                 query_thr=0.5, 
                 test_cfg=None, 
                 num_classes=200,
                 stuff_classes=[0, 1],
                 num_channels=32,
                 num_keep=200,
                 **kwargs):
        super().__init__()
    
        self._init_proj_layers()
        
        self.unet = SpConvUNet(
                num_planes=[num_channels * (i + 1) for i in range(5)],
                return_blocks=True)

        self.decoder = build_model(decoder)
        self.query_thr = query_thr
        self.test_cfg = test_cfg
        self.stuff_classes = stuff_classes

        self.thing_classes = np.array([i for i in range(num_classes) if i not in self.stuff_classes])
        
        self.num_inst_classes = len(self.thing_classes)
        
        self.num_keep = num_keep
    
    def _init_proj_layers(self):        
        self.input_conv = spconv.SparseSequential(
            SubMConv3d(
                6,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))

        self.output_layer = spconv.SparseSequential(
            BatchNorm1d(32, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))
        
        self.pos_encode = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.alignment_proj = nn.Sequential(
            nn.Linear(256, 1024),
        )

    def extract_feat(self, input_dict, sp_pts_masks, sp_batch_offsets):
        batch_size = len(input_dict["offset"])
        grid_coord = input_dict["grid_coord"]
        feats = input_dict["feat"]
        v2p_map = input_dict["v2p_map"]
        spatial_shape = input_dict["spatial_shape"]
        xyz_all = input_dict["coord"]

        # to align with the pretrained backbone
        feats = torch.cat([feats[:, 3:], feats[:, :3] - feats[:, :3].mean(0)], dim=1)
        compute_dtype = feats.dtype
        
        with torch.cuda.amp.autocast(enabled=False):
            voxel_feats = voxelization(feats.float(), v2p_map)
            voxel_input = spconv.SparseConvTensor(
                features=voxel_feats,
                indices=grid_coord.int(),
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )
        
        voxel_input = voxel_input.replace_feature(voxel_input.features.to(compute_dtype).detach())
        
        with torch.cuda.amp.autocast(enabled=False):
            voxel_input = self.input_conv(voxel_input)
            voxel_feats, _ = self.unet(voxel_input)
            voxel_feats = self.output_layer(voxel_feats)
            voxel_feats = voxel_feats.features

        # get point features from voxel features
        p2v_map = input_dict["p2v_map"].long()
        x = voxel_feats[p2v_map]
        x_pos = self.pos_encode(xyz_all)
        # get superpoint features from point features
        with torch.cuda.amp.autocast(enabled=False):
            x = scatter(x.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            x_pos = scatter(x_pos.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            sp_xyz = scatter(xyz_all.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            x = x + x_pos

        x = x.to(compute_dtype)
        sp_xyz = sp_xyz.to(compute_dtype)
        
        out = []
        out_xyz = []
        for i in range(len(sp_batch_offsets)-1):
            out.append(x[sp_batch_offsets[i]: sp_batch_offsets[i+1]])
            out_xyz.append(sp_xyz[sp_batch_offsets[i]: sp_batch_offsets[i+1]])
        
        return out, out_xyz
    
    def _select_queries(self, x, input_dict):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_sp_mask_list (List[Tensor]): of len batch_size.
                Groud truth of `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        gt_sp_mask_list = input_dict["gt_inst_sp_masks"]
        queries = []
        gt_query_mask_list = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_query_mask_list.append(gt_sp_mask_list[i][:, ids])
            else:
                queries.append(x[i])
                gt_query_mask_list.append(gt_sp_mask_list[i])
        return queries, gt_query_mask_list

    def predict_semantic(self, out_dict, sp_pts_masks, classes=None):
        if classes is None:
            classes = list(range(out_dict['sem_preds'][0].shape[1] - 1))
        return out_dict["sem_preds"][0][:, classes][sp_pts_masks]

    def predict_instance(self, out, sp_pts_masks, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        # score_threshold = self.test_cfg.inst_score_thr

        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_inst_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_inst_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, sp_pts_masks]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores


    def predict_top_queries(self, out, sp_pts_masks, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        # score_threshold = self.test_cfg.inst_score_thr

        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        num_classes = scores.shape[1]

        sorted_scores, indices = scores.sort(descending=True)
        indices = indices // num_classes
        _, unique_indices = indices.unique(return_inverse=True)

        topk_idx = indices[torch.unique_consecutive(unique_indices, return_counts=False)][:K]
        import pdb
        pdb.set_trace()


        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_inst_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_inst_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, sp_pts_masks]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores


    def predict_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_logits = self.predict_semantic(
            out, superpoints, self.stuff_classes)
        sem_map = sem_logits.argmax(dim=1)

        mask_pred, labels, scores  = self.predict_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map
        
        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)

        insts = inst_idxs * mask_pred

        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map


    def forward(self, input_dict, num_topk=100):
        batch_size = len(input_dict["offset"])
        sp_pts_mask_all = input_dict["superpoint_mask"]

        superpoint_bias = 0
        sp_pts_mask_list = []
        sp_batch_offsets = [0]
        for i in range(batch_size):
            sp_pts_mask = input_dict["superpoint_mask"][i]
            sp_pts_mask += superpoint_bias
            superpoint_bias += len(sp_pts_mask.unique())
            # superpoint_bias = sp_pts_mask.max().item() + 1
            sp_batch_offsets.append(superpoint_bias)
            sp_pts_mask_list.append(sp_pts_mask)
            
        sp_pts_masks = torch.hstack(sp_pts_mask_list)
        
        x, sp_xyz = self.extract_feat(input_dict, sp_pts_masks, sp_batch_offsets)

        queries = x
        out_dict = self.decoder(x, queries, sp_xyz)

        hidden_states = out_dict.pop("hidden_states")
        last_hidden_state = hidden_states[-1]
        aligned_sp_feat = [self.alignment_proj(cur_x) for cur_x in last_hidden_state]

        topk_query_feat = []
        topk_query_coord = []
        
        # keep the query with top-K scores
        for i in range(batch_size):
            with torch.no_grad():
                cls_preds = out_dict["cls_preds"][i]
                scores = F.softmax(cls_preds, dim=-1)[:, :-1]
                max_scores = scores.max(1)[0]
                num_keep = min(max_scores.shape[0], num_topk)
                _, topk_idx = max_scores.topk(num_keep, sorted=True)

            topk_query_feat.append(aligned_sp_feat[i][topk_idx])
            topk_query_coord.append(sp_xyz[i][topk_idx])

        return topk_query_feat, topk_query_coord, aligned_sp_feat, x, sp_xyz, hidden_states[:-1]