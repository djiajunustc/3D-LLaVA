# Modified from OneFormer3D

import torch
import torch.nn as nn
from .builder import MODELS


class CrossAttentionLayer(nn.Module):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why nn.Module doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(nn.Module):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


@MODELS.register_module()
class QueryDecoder(nn.Module):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder_SA_Only(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [0, 1, 2]
        self.cross_attn_layers = None
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.self_attn_layers)):
            # queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.self_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, xyz=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder_SA_Only_Dist_Bias(ScanNetQueryDecoder_SA_Only):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, d_model, num_layers, num_heads, dropout, **kwargs):
        super().__init__(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dropout=dropout, **kwargs)
        del self.input_proj 
        del self.cross_attn_layers 

        self.self_attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.self_attn_layers.append(
                DistBiasSelfAttention(d_model, num_heads, dropout))

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag and not self.training:
                sem_preds.append(self.out_sem(norm_query))
    
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
    
            if not self.training:
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
                pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks
    
    def forward_iter_pred(self, x, queries, xyz):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        hidden_states = []
        # inst_feats = [self.input_proj(y) for y in x]
        if not self.training:
            mask_feats = [self.x_mask(y) for y in x]
        else:
            mask_feats = None
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        hidden_states.append(queries)
        for i in range(len(self.self_attn_layers)):
            queries = self.self_attn_layers[i](queries, xyz)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.self_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            hidden_states.append(queries)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            hidden_states=hidden_states,
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, xyz=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, xyz)
        else:
            return self.forward_simple(x, queries)


class DistBiasSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)
        
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.scale_factor = nn.Parameter(torch.ones(num_heads) * 10, requires_grad=True)
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self, query_list, query_xyz_list):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        out = []
        for query_feat, query_xyz in zip(query_list, query_xyz_list):
            dist = self.calc_center_dists(query_xyz) # [Q, Q]
            tau = self.gen_tau(query_feat)  # [Q, 8]
            tau = tau.permute(1, 0)  # [8, Q]

            attn_mask = dist[None, :, :] * tau[..., None] * self.scale_factor[..., None, None]  # [8, Q, Q]
            # if pre_attn_mask is not None:
            #     attn_mask[:, :, pre_attn_mask] = float('-inf')
            # attn_mask = attn_mask.flatten(0, 1)  # [8, Q, Q]
            query_feat = query_feat.unsqueeze(0)
            new_query_feat, _ = self.attn(query_feat, query_feat, query_feat, attn_mask=attn_mask)
            query_feat = self.dropout(new_query_feat) + query_feat
            query_feat = self.norm(query_feat)
            query_feat = query_feat.squeeze(0)
            out.append(query_feat)
        return out

    @torch.no_grad()
    def calc_center_dists(self, centers):
        dist = torch.norm(centers.reshape(-1, 1, 3) - centers.reshape(1, -1, 3), dim=-1)
        dist = -dist

        return dist



@MODELS.register_module()
class VisualSampler_SA_Only_Dist_Bias(ScanNetQueryDecoder_SA_Only):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, d_model, num_layers, num_heads, dropout, **kwargs):
        super().__init__(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dropout=dropout, **kwargs)
        del self.input_proj 
        del self.cross_attn_layers 
        del self.out_norm
        del self.out_cls
        del self.x_mask
        del self.out_sem

        self.self_attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.self_attn_layers.append(
                DistBiasSelfAttention(d_model, num_heads, dropout))

    def forward(self, prompt_mask, multilevel_x, xyz=None):
        batch_size = len(prompt_mask)
        prompt_x = []
        prompt_count = 0
        for batch_ind in range(batch_size):
            cur_mask = prompt_mask[batch_ind]
            cur_xyz  = xyz[batch_ind]
            if len(cur_mask) == 0:
                cur_prompt_x = multilevel_x[0][batch_ind][0:0]
                cur_prompt_xyz = xyz[batch_ind].new_zeros((0, 3))
                offset = 0
            else:
                cur_prompt_x = torch.mean(multilevel_x[0][batch_ind][cur_mask], dim=0, keepdim=True)
                cur_prompt_xyz = torch.mean(cur_xyz[cur_mask], dim=0, keepdim=True)
                prompt_count += 1
                offset = 1
            cur_xyz = torch.cat([cur_prompt_xyz, cur_xyz], dim=0)
            for i in range(len(self.self_attn_layers)):
                cur_x = multilevel_x[i][batch_ind].detach()
                cur_x = torch.cat([cur_prompt_x, cur_x], dim=0)
                
                cur_x = self.self_attn_layers[i]([cur_x], [cur_xyz])
                cur_prompt_x = cur_x[0][0:offset]
                cur_prompt_x = self.ffn_layers[i]([cur_prompt_x])[0]
            prompt_x.append(cur_prompt_x)
        return prompt_x



@MODELS.register_module()
class MaskDecoder_SA_Only(ScanNetQueryDecoder_SA_Only):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, d_model, num_layers, num_heads, dropout, **kwargs):
        super().__init__(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dropout=dropout, **kwargs)
        del self.input_proj 
        del self.cross_attn_layers 
        del self.out_cls
        del self.out_sem

        self.self_attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))

    def _forward_head(self, queries, mask_feats):
        pred_masks, pred_scores = [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            pred_masks.append(pred_mask)
            pred_scores.append(None)
        return pred_masks, pred_scores
        

    def forward(self, seg_embeds, mask_x, hidden_states):
        batch_size = len(seg_embeds)
        pred_masks, pred_scores = [], []
        mask_feats = [self.x_mask(y) for y in mask_x]
        queries = seg_embeds
        offsets = [q.shape[0] for q in queries]

        pred_mask, pred_score = self._forward_head(queries, mask_feats)
        pred_masks.append(pred_mask)
        pred_scores.append(pred_score)

        for i in range(len(self.self_attn_layers)):
            x = hidden_states[i]
            for batch_ind in range(batch_size):
                x[batch_ind] = torch.cat([queries[batch_ind], x[batch_ind]], dim=0)
            x = self.self_attn_layers[i](x)
            queries = [cur_x[0:offset] for cur_x, offset in zip(x, offsets)]
            queries = self.ffn_layers[i](queries)
            pred_mask, pred_score = self._forward_head(queries, mask_feats)
            pred_masks.append(pred_mask)
            pred_scores.append(pred_score)
        
        aux_outputs = [{'masks': masks, 'scores': scores}
            for scores, masks in zip(
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            masks=pred_masks[-1], 
            scores=pred_scores[-1],
            aux_outputs=aux_outputs
            )
