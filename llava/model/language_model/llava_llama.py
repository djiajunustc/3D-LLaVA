#    Modified from LLaVA
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import (
    replace_return_docstrings,
)
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward_mask_decoder_train(
            self, 
            llm_out, 
            mask_input_dict, 
            seg_token_mask, 
            gt_seg_labels, 
            gt_seg_masks,
            conditions
            ):
        
        sp_features = mask_input_dict["sp_features"]
        pc_hidden_states = mask_input_dict["hidden_states"]
        batch_size = len(sp_features)

        llm_hidden_states = llm_out.hidden_states
        last_llm_hidden_states = llm_hidden_states[-1]
        seg_embeds = []
        for batch_ind in range(batch_size):
            cur_hidden_embeds = last_llm_hidden_states[batch_ind]
            cur_seg_token_mask = seg_token_mask[batch_ind]
            cur_seg_hidden_state = cur_hidden_embeds[cur_seg_token_mask]
            cur_seg_embed = self.get_hidden_seg_fc()(cur_seg_hidden_state)
            seg_embeds.append(cur_seg_embed)

        pred_dict = self.get_mask_decoder()(seg_embeds, sp_features, pc_hidden_states)

        seg_loss = self.get_seg_criteria()(
            pred_dict=pred_dict,
            query_masks=None,
            gt_masks=gt_seg_masks,
            gt_labels=gt_seg_labels,
        )
        llm_out["loss"] += seg_loss

        return llm_out

    def forward_mask_decoder_test(
            self, 
            mask_input_dict, 
            seg_hidden_state,
            superpoint_mask,
            conditions
            ):
        
        condition = conditions[0]
        assert condition == "refer_seg"
        seg_embeds = [self.get_hidden_seg_fc()(emb) for emb in seg_hidden_state]
        features_for_seg = mask_input_dict["sp_features"]
        hidden_states_for_seg = mask_input_dict["hidden_states"]

        pred_dict = self.get_mask_decoder()(seg_embeds, features_for_seg, hidden_states_for_seg)

        assert len(pred_dict['masks']) == 1
        pred_masks = pred_dict['masks'][0]
        p2s_masks  = superpoint_mask[0]
        
        pred_superpoint_seg_mask = pred_masks

        pred_superpoint_seg_mask = pred_superpoint_seg_mask.sigmoid() > 0.5
        pred_seg_mask = pred_superpoint_seg_mask[:, p2s_masks]
        return pred_seg_mask

    @replace_return_docstrings(output_type=CausalLMOutputWithPast)
    def llama_inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)[...,:-1]
        seg_logit = self.get_model().lm_head_seg(hidden_states)
        logits = torch.cat([logits, seg_logit], dim=-1)

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # added ones start here
        coord: Optional[torch.FloatTensor] = None,
        grid_coord: Optional[torch.LongTensor] = None,
        offset: Optional[torch.LongTensor] = None,
        feat: Optional[torch.FloatTensor] = None,
        p2v_map: Optional[torch.IntTensor] = None,
        v2p_map: Optional[torch.IntTensor] = None,
        spatial_shape: Optional[torch.LongTensor] = None,
        conditions: Optional[List] = None,
        gt_seg_masks: Optional[List] = None,
        gt_seg_labels: Optional[List] = None,
        superpoint_mask: Optional[List] = None,
        click_mask: Optional[List] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                seg_token_mask,
                mask_input_dict
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                coord,
                grid_coord,
                offset,
                feat,
                p2v_map,
                v2p_map,
                spatial_shape,
                superpoint_mask,
                click_mask,
            )

        
        llm_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        
        if not self.training:
            return llm_out

        else:
            out = self.forward_mask_decoder_train(
                llm_out, 
                mask_input_dict, 
                seg_token_mask, 
                gt_seg_labels, 
                gt_seg_masks,
                conditions
            )

            return out

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        coord: Optional[torch.FloatTensor] = None,
        grid_coord: Optional[torch.LongTensor] = None,
        offset: Optional[torch.LongTensor] = None,
        feat: Optional[torch.FloatTensor] = None,
        p2v_map: Optional[torch.IntTensor] = None,
        v2p_map: Optional[torch.IntTensor] = None,
        spatial_shape: Optional[torch.LongTensor] = None,
        superpoint_mask: Optional[List] = None,
        tokenizer=None,
        click: Optional[torch.FloatTensor] = None,
        click_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or coord is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                mask_input_dict
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                coord,
                grid_coord,
                offset,
                feat,
                p2v_map,
                v2p_map,
                spatial_shape,
                superpoint_mask,
                click_mask=click_mask
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )
        output_hidden_states = outputs.hidden_states
        output_ids = outputs.sequences
        conditions = kwargs.pop("conditions")
        if conditions[0] in ["vqa", "captioning", "textgen"]:
            return output_ids

        assert output_ids.shape[0] == 1, "only support batch_size=1"
        seg_token_mask = output_ids[0, 1:] == self.config.seg_token_idx
        
        seg_indices = torch.nonzero(seg_token_mask, as_tuple=True)[0] 
        # if without [SEG] token, output all zeros mask
        if len(seg_indices) == 0:
            empty_mask = torch.zeros(coord.shape[0]).to(coord.device)
            return empty_mask

        seg_hidden_states = [output_hidden_states[i] for i in seg_indices]
        last_seg_hidden_states = [h[-1] for h in seg_hidden_states]
        last_seg_hidden_states = torch.cat(last_seg_hidden_states, dim=1)
        last_seg_hidden_states = list(last_seg_hidden_states)

        pred_mask = self.forward_mask_decoder_test(
            mask_input_dict,
            seg_hidden_state=last_seg_hidden_states,
            superpoint_mask=superpoint_mask,
            conditions=conditions
            )

        return pred_mask

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):

        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        return inputs

    
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
