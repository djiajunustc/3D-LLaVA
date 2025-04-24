#    Modified from LLaVA repository
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import (build_vision_tower, 
                                         build_pointcloud_tower, 
                                         build_prompt_encoder,
                                         build_inst_prompt_encoder)
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, LOC_TOKEN_INDEX

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            if config.mm_vision_tower is not None:
                self.vision_tower = build_vision_tower(config, delay_load=True)
                self.mm_projector = build_vision_projector(config)

                if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                    self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )
        
        if hasattr(config, "mm_pointcloud_tower"):
            self.pointcloud_tower = build_pointcloud_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

        
        if hasattr(config, "mm_inst_prompt_encoder"):
            if not config.mm_inst_prompt_encoder == "shared_projector":
                self.inst_prompt_encoder = build_inst_prompt_encoder(config)
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_pointcloud_tower(self):
        pointcloud_tower = getattr(self, 'pointcloud_tower', None)
        if type(pointcloud_tower) is list:
            pointcloud_tower = pointcloud_tower[0]
        return pointcloud_tower
    
    def get_mask_decoder(self):
        pointcloud_tower = self.get_pointcloud_tower()
        mask_decoder = pointcloud_tower.mask_decoder
        return mask_decoder

    def get_hidden_seg_fc(self):
        pointcloud_tower = self.get_pointcloud_tower()
        hidden_seg_fc = pointcloud_tower.hidden_seg_fc
        return hidden_seg_fc

    def get_seg_criteria(self):
        pointcloud_tower = self.get_pointcloud_tower()
        seg_criteria = pointcloud_tower.seg_criteria
        return seg_criteria

    def get_prompt_encoder(self):
        prompt_encoder = getattr(self, 'prompt_encoder', None)
        return prompt_encoder

    def get_inst_prompt_encoder(self):
        inst_prompt_encoder = getattr(self, 'inst_prompt_encoder', None)
        return inst_prompt_encoder

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_pc_mlp_adapter = model_args.pretrain_pc_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        # newly added ones
        pointcloud_tower = model_args.pointcloud_tower
        pointcloud_decoder = model_args.pointcloud_decoder
        prompt_encoder = model_args.prompt_encoder
        inst_prompt_encoder = model_args.inst_prompt_encoder


        self.config.mm_vision_tower = vision_tower
        if vision_tower is not None:
            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(vision_tower, 'hidden_size', 1024)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        self.config.mm_pointcloud_tower = pointcloud_tower
        if pointcloud_tower is not None:
            if self.get_pointcloud_tower() is None:
                pointcloud_tower = build_pointcloud_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.pointcloud_tower = [pointcloud_tower]
                else:
                    self.pointcloud_tower = pointcloud_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    pointcloud_tower = self.pointcloud_tower[0]
                else:
                    pointcloud_tower = self.pointcloud_tower
                pointcloud_tower.load_model()

        self.config.use_pc_proj = True
        self.config.pc_hidden_size = pointcloud_tower.hidden_size
        self.config.pc_feature_dim = pointcloud_tower.feature_dim

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        # build visual sampler
        self.config.mm_inst_prompt_encoder = inst_prompt_encoder
        if inst_prompt_encoder is not None:
            if self.get_inst_prompt_encoder() is None:
                if self.config.mm_inst_prompt_encoder not in ["shared_projector"]:
                #     self.inst_prompt_encoder = pointcloud_tower.alignment_proj
                # else:
                    self.inst_prompt_encoder = build_inst_prompt_encoder(self.config)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_pointcloud_tower(self):
        return self.get_model().get_pointcloud_tower()

    def get_pointcloud_decoder(self):
        return self.get_model().get_pointcloud_decoder()

    def get_mask_decoder(self):
        return self.get_model().get_mask_decoder()

    def get_hidden_seg_fc(self):
        return self.get_model().get_hidden_seg_fc()

    def get_seg_criteria(self):
        return self.get_model().get_seg_criteria()
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_pointclouds(
        self, coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask
        ):
        sampled_features, prompt_features, superpoint_features, mask_input_dict = self.get_model().get_pointcloud_tower()(
            coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask
        )

        pointcloud_tokens = [self.get_model().mm_projector(feat) for feat in sampled_features]
        prompt_tokens = [self.get_model().mm_projector(feat) for feat in prompt_features]
    
        return pointcloud_tokens, prompt_tokens, superpoint_features, mask_input_dict

    def encode_click_prompt(self, prompt):
        prompt_tokens = self.get_model().get_prompt_encoder()(prompt)
        return prompt_tokens

    def encode_inst_prompt(self, feature):
        if self.config.mm_inst_prompt_encoder == "shared_projector":
            # prompt_tokens = self.get_model().pointcloud_tower.alignment_proj(feature)
            prompt_tokens = self.get_model().mm_projector(feature)
        else:
            prompt_tokens = self.get_model().get_inst_prompt_encoder()(feature)
        return prompt_tokens
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, 
        labels=None, coord=None, grid_coord=None, offset=None, pc_input=None, 
        p2v_map=None, v2p_map=None, spatial_shape=None, superpoint_mask=None,
        click_mask=None,
    ):  
        pointcloud_tower = self.get_pointcloud_tower()
        if pointcloud_tower is None or pc_input is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        
        pc_tokens, prompt_tokens, superpoint_features, mask_input_dict = self.encode_pointclouds(
             coord, grid_coord, offset, pc_input, p2v_map, v2p_map, spatial_shape, superpoint_mask, click_mask
        )

        prompt_tokens = torch.cat(prompt_tokens, dim=0)

        # get segmentation token index, shift 1 token since the prediciton is the next token
        seg_token_mask = input_ids[:, 1:] == self.config.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention mask
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        seg_token_mask = [cur_seg_mask[cur_attention_mask] for cur_seg_mask, cur_attention_mask in zip(seg_token_mask, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_seg_token_mask = []
        # Use "image" to represent "3d scene" in the following code, following the practice of LLaVA
        # Remember that in the current implementation, the <pc> also uses IMAGE_TOKEN_INDEX
        # May by modified in the future version
        cur_image_idx = 0
        cur_prompt_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_prompts = (cur_input_ids == LOC_TOKEN_INDEX).sum()
            num_specials = num_images + num_prompts
            if num_images == 0:
                cur_pc_features = pc_tokens[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # it seems that the following code has not actually invovled cur_pc_featurs?
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_pc_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            prompt_token_indices = torch.where(cur_input_ids == LOC_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + prompt_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_seg_token_mask = seg_token_mask[batch_idx]
            cur_seg_token_mask_noim = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_seg_token_mask_noim.append(cur_seg_token_mask[special_token_indices[i]+1:special_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_seg_token_mask = []
            for i in range(num_specials + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_seg_token_mask.append(cur_seg_token_mask_noim[i])
                if i < num_specials:
                    cur_token = special_tokens[i]
                    if cur_token == IMAGE_TOKEN_INDEX:
                        cur_pc_features = pc_tokens[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_pc_features)
                        cur_new_labels.append(torch.full((cur_pc_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_seg_token_mask.append(torch.full((cur_pc_features.shape[0],), False, device=cur_seg_token_mask.device, dtype=cur_seg_token_mask.dtype))
                    elif cur_token == LOC_TOKEN_INDEX:
                        cur_prompt_features = prompt_tokens[cur_prompt_idx]
                        if len(cur_prompt_features.shape) == 1:
                            cur_prompt_features = cur_prompt_features.unsqueeze(0)
                        cur_prompt_idx += 1
                        cur_new_input_embeds.append(cur_prompt_features)
                        cur_new_labels.append(torch.full((cur_prompt_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_seg_token_mask.append(torch.full((cur_prompt_features.shape[0],), False, device=cur_seg_token_mask.device, dtype=cur_seg_token_mask.dtype))


            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_seg_token_mask = torch.cat(cur_new_seg_token_mask)

            if num_prompts == 0:
                cur_new_input_embeds = torch.cat([cur_new_input_embeds, prompt_tokens[0:0]], dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_seg_token_mask.append(cur_new_seg_token_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_seg_token_mask = [x[:tokenizer_model_max_length] for x in new_seg_token_mask]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        seg_token_mask_padded = torch.full((batch_size, max_len), False, dtype=new_seg_token_mask[0].dtype, device=new_seg_token_mask[0].device)

        for i, (cur_new_embed, cur_new_labels, cur_new_seg_token_mask) in \
                    enumerate(zip(new_input_embeds, new_labels, new_seg_token_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    seg_token_mask_padded[i, -cur_len:] = cur_new_seg_token_mask
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    seg_token_mask_padded[i, :cur_len] = cur_new_seg_token_mask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, seg_token_mask_padded, mask_input_dict

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
