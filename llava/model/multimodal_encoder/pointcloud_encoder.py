import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_geometric.utils import scatter
from .custom_spconv_module.spconv_layers import SubMConv3d
from .custom_spconv_module.batchnorm import BatchNorm1d
import spconv.pytorch as spconv
import pathlib
from llava.utils import load_config
from .ost_module import build_model, build_criteria
import copy


class SPConvPointCloudTower(nn.Module):
    def __init__(self, pointcloud_tower, args, hidden_dim=32, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.pointcloud_tower_name = pointcloud_tower
        self.num_pc_tokens = args.num_pc_tokens
        if "llava_distill" in self.pointcloud_tower_name or \
            "align" in self.pointcloud_tower_name:
            self.hidden_dim = 1024
        else:
            self.hidden_dim = hidden_dim

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_pointcloud_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None, pointcloud_tower_name=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return
        
        if pointcloud_tower_name is not None:
            self.pointcloud_tower_name = pointcloud_tower_name

        config_name = self.pointcloud_tower_name.split('/')[-1]
        config_name = config_name.split('.')[0]

        current_file_path = pathlib.Path(__file__).resolve()
        current_dir = current_file_path.parent
        config_path = current_dir / 'configs' / f'{config_name}.py'
        cfg = load_config(config_path)

        self.segmentor = build_model(cfg.model)

        self.load_segmentor_weights()
        
        self.alignment_proj = self.segmentor.alignment_proj

        # re-use OST as the visual sampler
        self.visual_sampler = build_model(cfg.visual_sampler)

        self.visual_sampler.load_state_dict(
            copy.deepcopy(self.segmentor.decoder.state_dict()), 
            strict=False
            )

        for p in self.visual_sampler.parameters():
            p.requires_grad = False

        # re-use OST as the mask decoder
        self.mask_decoder = build_model(cfg.mask_decoder)

        self.mask_decoder.load_state_dict(
            copy.deepcopy(self.segmentor.decoder.state_dict()), 
            strict=False
            )

        for p in self.mask_decoder.parameters():
            p.requires_grad = True

        # projection layer for [SEG] tokens
        seg_fc = [
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.0),
        ]
        self.hidden_seg_fc = nn.Sequential(*seg_fc)
        self.hidden_seg_fc.train()
        for param in self.hidden_seg_fc.parameters():
            param.requires_grad = True

        # segmentation criterion
        self.seg_criteria = build_criteria(cfg.criteria)

        self.is_loaded = True
    
    def load_segmentor_weights(self):
        ckpt_dir = self.pointcloud_tower_name.replace("_fp32", "")
        checkpoint = torch.load(ckpt_dir, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        checkpoint_to_load = dict()

        for name, val in checkpoint.items():
            checkpoint_to_load[name.replace("module.", "")] = val
        missing, unexpected = self.segmentor.load_state_dict(checkpoint_to_load, strict=False)

    def forward(self, coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask):
        batch_size = len(offset)

        segmentor_input_dict = dict(
            coord=coord,
            grid_coord=grid_coord,
            offset=offset,
            feat=feat,
            p2v_map=p2v_map,
            v2p_map=v2p_map,
            spatial_shape=spatial_shape,
            superpoint_mask=superpoint_mask,
        )
        
        topk_query_feat, topk_query_coord, aligned_sp_feat, sp_feature, sp_xyz, hidden_states = self.segmentor(segmentor_input_dict, num_topk=self.num_pc_tokens)
        
        prompt_feature = self.visual_sampler(prompt_mask, hidden_states, sp_xyz)
        prompt_feature = [self.alignment_proj(ff) for ff in prompt_feature]
        
        mask_input_dict = {
            "sp_features": sp_feature,
            "hidden_states": hidden_states
        }
        return topk_query_feat, prompt_feature, aligned_sp_feat, mask_input_dict
        

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.unet.dtype

    @property
    def device(self):
        return self.unet.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.hidden_dim

    @property
    def feature_dim(self):
        return self.hidden_dim