# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import tokenizers
import numpy as np

from collections.abc import Mapping, Sequence
from pointgroup_ops import voxelization_idx

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# added tokens
from llava.constants import DEFAULT_PC_TOKEN, DEFAULT_LINK_TOKEN, LINK_TOKEN_INDEX
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, tokenizer_special_token
from llava.pc_utils import (Compose, 
                            referseg_transform_train, 
                            vqa_transform_train,
                            densecap_transform_train)

from PIL import Image
from collections import defaultdict
import random
import wandb
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # new added arguments for point cloud inputs
    pointcloud_tower: Optional[str] = field(default=None)
    freeze_pointcloud_tower: bool = field(default=False)
    freeze_pointcloud_decoder: bool = field(default=False)
    tune_pc_mlp_adapter: bool = field(default=False)
    tune_mask_decoder: bool = field(default=True)
    pc_use_link_token: bool = field(default=False)
    pretrain_pc_mlp_adapter: Optional[str] = field(default=None)
    pc_projector_type: Optional[str] = field(default='linear')
    pointcloud_decoder: Optional[str] = field(default=None)
    pc_sampling_type: Optional[str] = field(default='superpoint')
    num_pc_tokens: int = 256
    num_link_tokens: int = 64
    prompt_encoder: Optional[str] = field(default="click_prompt")
    inst_prompt_encoder: Optional[str] = field(default="inst_feature_prompt")

@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

    # new added arguments for point cloud inputs
    scan_folder: Optional[str] = field(default=None)
    scene_wise: bool = False
    scene_alignment_file: Optional[str] = field(default=None)
    extra_det_file: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    pc_projector_lr: Optional[float] = None
    pointcloud_tower_lr: Optional[float] = None
    pointcloud_decoder_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_task_length: bool = field(default=False)
    group_by_task_ratio: bool=field(default=False)
    group_by_task_length_per_batch: bool=field(default=False)
    # new added options
    steps_per_epoch: int = -1
    pretrained_checkpoint: Optional[str] = None
    pc_modules_to_finetune: Optional[List[str]] = field(default=None,
                           metadata={"help": "keywords of modules to be finetuned."})

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, named_buffers, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    
    buffer_dict = {k: t for k, t in named_buffers if "lora_" not in k}
    buffer_dict = {k: t for k, t in buffer_dict.items() if "running_mean" in k or "running_var" in k}
    to_return.update(buffer_dict)  # add buffers, linke running mean, var for bn

    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}

    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'pc_projector', 'pointcloud_tower', 'pointcloud_decoder']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'lm_head_seg', 'hidden_seg_fc']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        
        # Save other paramters
        other_keys_to_match = ['lm_head_seg', 'hidden_seg_fc']
        
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_PC_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_PC_TOKEN, DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

            replace_token, replace_pc_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    add_link_token: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if role == conv.roles[0] and add_link_token:
                sentence["value"] = sentence["value"] + "\n" + DEFAULT_LINK_TOKEN
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_special_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_special_token(rou, tokenizer))
                instruction_len = len(tokenizer_special_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    add_link_token: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, add_link_token=add_link_token)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        list_data_dict = []
        for dataset_idx, cur_data_path in enumerate(data_path):
            cur_list_data_dict = json.load(open(cur_data_path, "r"))
            for cur_data in cur_list_data_dict:
                cur_data["id"] = "%02d"%dataset_idx + "_{}".format(cur_data["id"])
                cur_data["scene_id"] = "d%02d-"%dataset_idx + cur_data["scene_id"]
                list_data_dict.append(cur_data)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.cache = {}  # Initialize the cache for scan_data_path

        if self.data_args.scene_wise:
            unq_scenes = sorted(list(set([data['scene_id'] for data in list_data_dict])))
            self.list_data_dict = [{'scene_id': scene_id} for scene_id in unq_scenes]
            scene2conv_dict = defaultdict(list)
            for data in list_data_dict:
                scene_id = data['scene_id']
                scene2conv_dict[scene_id].append(data)
            self.scene2conv_dict = scene2conv_dict
        else:
            self.list_data_dict = list_data_dict
        self._build_pc_transform()

    def _build_pc_transform(self):
        self.rs_transform = Compose(referseg_transform_train)

        if self.data_args.scene_alignment_file is not None:
            vqa_transform_train[0]['file_path'] = self.data_args.scene_alignment_file
            densecap_transform_train[0]['file_path'] = self.data_args.scene_alignment_file

        self.vqa_transform = Compose(vqa_transform_train)
        self.densecap_transform = Compose(densecap_transform_train)

    def __len__(self):
        if self.data_args.samples_per_epoch > 0:
            return self.data_args.samples_per_epoch
        else:
            return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample or 'scene_id' in sample:
                cur_len = cur_len 
            else:
                cur_len = -cur_len
            
            # to seperate text-gen and segmentation samples
            if sample['task_type'] in ['refer_seg']: 
                cur_len += 1000
            length_list.append(cur_len)

        return length_list

    @property
    def task_lengths(self):
        task_mapping = {
            "vqa": 0,
            "dense_captioning": 1,
            "refer_seg": 2,
        }
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            length_list.append(
                (task_mapping[sample['task_type'].lower()], cur_len)
            )
        return length_list

    def load_and_cache_scan_data(self, scan_data_path, superpoint_path):
        """Helper function to load and cache scan data."""
        if scan_data_path in self.cache:
            # If the data is already cached, return it from the cache
            return self.cache[scan_data_path]

        # Otherwise, load the data from disk
        raw_data = torch.load(scan_data_path)
        coord = raw_data['coord']
        color = raw_data['color']
        segment = raw_data['semantic_gt20']
        instance = raw_data['instance_gt']
        superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

        # Store the loaded data in the cache
        self.cache[scan_data_path] = {
            'coord': coord,
            'color': color,
            'segment': segment,
            'instance': instance,
            'superpoint_mask': superpoint_mask
        }

        return self.cache[scan_data_path]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.data_args.samples_per_epoch > 0:
            i = random.randint(0, len(self.list_data_dict)-1)
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        elif 'scene_id' in sources[0]:
            scene_id = sources[0]['scene_id']
            scene_name = scene_id.split('-')[-1]
            scan_folder = self.data_args.scan_folder
            scan_data_path  = pathlib.Path(scan_folder) / 'train' / f'{scene_name}.pth'
            superpoint_path = pathlib.Path(scan_folder) / 'super_points' / f'{scene_name}.bin'
            # load point cloud data
            # raw_data = torch.load(scan_data_path)
            # superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)
            raw_data = copy.deepcopy(self.load_and_cache_scan_data(scan_data_path, superpoint_path))
            coord = raw_data['coord']
            color = raw_data['color']
            segment = raw_data['segment']
            instance = raw_data['instance']
            superpoint_mask = raw_data['superpoint_mask']

            if self.data_args.scene_wise:
                chosen_conv = random.choice(self.scene2conv_dict[scene_id])
                conv_sources = [copy.deepcopy(chosen_conv)]
            else:
                conv_sources = sources
            
            if 'task_type' in conv_sources[0].keys():
                task_type = conv_sources[0]['task_type']
            else:
                task_type = 'refer_seg'

            if task_type == 'refer_seg':
                transform = self.rs_transform
            elif task_type == 'vqa':
                transform = self.vqa_transform
            elif task_type == 'dense_captioning':
                transform = self.densecap_transform

            pc_data_dict = dict(
                scene_id=scene_name,
                coord=coord,
                color=color,
                segment=segment,
                instance=instance,
                superpoint_mask=superpoint_mask,
                conversation=copy.deepcopy(conv_sources[0]["conversations"])
            )

            if "object_id" in conv_sources[0]:
                object_id = conv_sources[0]['object_id']
                pc_data_dict["object_id"] = object_id

            if "pred_id" in conv_sources[0]:
                pc_data_dict["pred_id"] = conv_sources[0]["pred_id"]
            
            pc_data_dict = transform(pc_data_dict)
            targets = [e.get("target", None) for e in conv_sources]
            
            conv_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in conv_sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            conv_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i] or 'scene_id' in self.list_data_dict[i]),
            add_link_token=self.data_args.pc_use_link_token)
    
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'scene_id' in self.list_data_dict[i]:
            data_dict['scene_id'] = self.list_data_dict[i]['scene_id']
            data_dict.update(pc_data_dict)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


def ponder_collate_fn(batch, max_point=-1):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    # we drop a large data if it exceeds max_point
    # note that directly drop the last one may cause problem
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for batch_id, data in enumerate(batch):
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'scene_id' in instances[0]:
            batch_size = len(instances)
            if "grid_coord" in instances[0].keys():
                for i in range(batch_size):
                    grid_coord = instances[i]["grid_coord"]
                    grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(i), grid_coord], 1)
                    instances[i]["grid_coord"] = grid_coord
            
            for key in instances[0]:
                if key in ["coord", "grid_coord", "feat", "offset"]:
                    batch[key] = ponder_collate_fn([d[key] for d in instances])
            
            for key in batch.keys():
                if "offset" in key:
                    batch[key] = torch.cumsum(batch[key], dim=0)
            
            # voxelize
            grid_coords = batch["grid_coord"]
            spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
            voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

            # other items in input, given as list
            scene_id_list = []
            condition_list = []
            gt_mask_list = []
            gt_label_list = []
            superpoint_mask_list = []
            click_list = []
            click_mask_list = []
            for d in instances:
                scene_id_list.append(d["scene_id"])
                condition_list.append(d["condition"])
                if "gt_masks_3d" in d:
                    gt_mask_list.append(d["gt_masks_3d"])
                else:
                    gt_mask_list.append([])
                if "gt_labels_3d" in d:
                    gt_label_list.append(d["gt_labels_3d"])
                else:
                    gt_label_list.append([])
                if 'obj_click' in d:
                    click_list.append(d['obj_click'])
                if 'obj_sp_mask' in d:
                    click_mask_list.append(d['obj_sp_mask'])
                else:
                    click_mask_list.append([])
                superpoint_mask_list.append(d["superpoint_mask"])
            if len(click_list) > 0:
                click_tensor = torch.stack(click_list)
            else:
                click_tensor = None
            batch.update({
                "grid_coord": voxel_coords,
                "p2v_map": p2v_map,
                "v2p_map": v2p_map,
                "spatial_shape": spatial_shape,
                "conditions": condition_list,
                "gt_seg_masks": gt_mask_list,
                "gt_seg_labels": gt_label_list,
                "superpoint_mask": superpoint_mask_list,  
                "scene_id": scene_id_list,
                "click": click_tensor,
                "click_mask": click_mask_list
            })
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None or model_args.pointcloud_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None or model_args.pointcloud_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        if model_args.vision_tower is not None:
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True
        
        if model_args.pointcloud_tower is not None:
            pointcloud_tower = model.get_pointcloud_tower()
            pointcloud_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.is_multimodal = True
    
        if model_args.pointcloud_decoder is not None:
            pointcloud_decoder = model.get_pointcloud_decoder()
            pointcloud_decoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            if model_args.freeze_pointcloud_decoder:
                for p in model.get_model().pointcloud_decoder.parameters():
                    p.requires_grad = False
            else:
                for p in model.get_model().pointcloud_decoder.parameters():
                    p.requires_grad = True
                    
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # add special tokens to tokenizer
        if "[SEG]" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': ["[SEG]"]})
            model.resize_token_embeddings(len(tokenizer))
            
        seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
        # model.lm_head.weight[seg_token_idx] = 0.0
        model_args.seg_token_idx = seg_token_idx
        model.config.seg_token_idx = model_args.seg_token_idx
        model.config.link_token_indices = -1000000

        # enable gradient bp for token_embeds and lm_head
        for p in model.lm_head.parameters():
            p.requires_grad = True
        for p in model.get_model().embed_tokens.parameters():
            p.requires_grad = True

        assert model_args.freeze_pointcloud_tower, "Only support freeze pointcloud tower"

        if model_args.freeze_pointcloud_tower:
            # pc_tower_finetune_keywords = [
            #     "alignment_proj", "scene_transformer", "mask_decoder", "hidden_seg_fc"
            # ]
            pc_tower_finetune_keywords = training_args.pc_modules_to_finetune
            if pc_tower_finetune_keywords is None:
                pc_tower_finetune_keywords = []
            for name, param in model.get_model().pointcloud_tower.named_parameters():
                if not any(keyword in name for keyword in pc_tower_finetune_keywords):
                    param.requires_grad = False
        else:
            for p in model.get_model().pointcloud_tower.parameters():
                p.requires_grad = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            for p in model.get_model().lm_head_seg.parameters():
                p.requires_grad = True
            hidden_seg_fc = model.get_model().get_hidden_seg_fc()
            for p in hidden_seg_fc.parameters():
                p.requires_grad = True
            alignment_proj = model.get_model().get_pointcloud_tower().alignment_proj
            for p in alignment_proj.parameters():
                p.requires_grad = True
                
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.pc_projector_lr = training_args.pc_projector_lr
        model.config.pointcloud_tower_lr = training_args.pointcloud_tower_lr
        model.config.pointcloud_decoder_lr = training_args.pointcloud_decoder_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        # set number of sampled superpoint features as vision tokens
        model.config.num_pc_tokens = model_args.num_pc_tokens
        model.config.pc_sampling_type = model_args.pc_sampling_type
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_args.pc_use_link_token = model_args.pc_use_link_token
    if training_args.steps_per_epoch > 0:
        world_size = torch.cuda.device_count()
        data_args.samples_per_epoch = training_args.per_device_train_batch_size * \
                                      world_size * \
                                      training_args.gradient_accumulation_steps * \
                                      training_args.steps_per_epoch
    else:
        data_args.samples_per_epoch = -1
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if training_args.pretrained_checkpoint:
        trainable_record = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_record.append(name) 

        rank0_print("Use pretrained checkpoint with LoRA params and non_lora_trainable params...")
        model_path = training_args.pretrained_checkpoint
        
        rank0_print('Loading LoRA weghts...')
        if os.path.exists(os.path.join(model_path, 'adapter_model.bin')):
            lora_dict = torch.load(os.path.join(model_path, 'adapter_model.bin'), map_location=training_args.device)
            lora_dict = {k.replace('weight', 'default.weight'): v for k, v in lora_dict.items()}
            missing_keys, unexpected_keys = model.load_state_dict(lora_dict, strict=False)
        
        rank0_print('Loading additional 3D-LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location=training_args.device)
            non_lora_trainables = {k:v for k, v in non_lora_trainables.items() if 'mm_projector' not in k}
            model.load_state_dict(non_lora_trainables, strict=False)

        for name, param in model.named_parameters():
            if name in trainable_record:
                param.to(training_args.device)
                param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), model.named_buffers()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
