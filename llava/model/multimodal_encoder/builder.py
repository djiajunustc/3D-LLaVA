import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .pointcloud_encoder import SPConvPointCloudTower
from .prompt_encoder import InstPromptEncoder
from .pointcloud_encoder import SPConvPointCloudTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_pointcloud_tower(pc_tower_cfg, **kwargs):
    
    pc_tower = getattr(pc_tower_cfg, 'mm_pointcloud_tower', getattr(pc_tower_cfg, 'pointcloud_tower', None))

    return SPConvPointCloudTower(pc_tower, args=pc_tower_cfg, **kwargs)


def build_prompt_encoder(pc_tower_cfg):

    return PromptEncoder(hidden_size=pc_tower_cfg.hidden_size)


def build_inst_prompt_encoder(pc_tower_cfg):

    return InstPromptEncoder(
        input_dim=pc_tower_cfg.pc_feature_dim, hidden_size=pc_tower_cfg.hidden_size
        )