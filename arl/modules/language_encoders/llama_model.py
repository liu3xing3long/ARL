# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

import copy
import math
import os
import warnings
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

logger = logging.get_logger(__name__)


class LlamaCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LlamaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = LlamaAttention(config)
        self.intermediate = LlamaDecoderLayer(config)
        self.output = LlamaMLP(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None  # past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]
        # outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LlamaCrossLayerWithKnowledge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LlamaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = LlamaAttention(config)
        self.intermediate = LlamaDecoderLayer(config)
        self.output = LlamaMLP(config)

        self.knowledge_config = copy.deepcopy(config)
        self.knowledge_config.hidden_size = 256
        self.knowledge_config.intermediate_size = 1024
        self.knowledge_config.num_attention_heads = 4

        self.knowledge_attention = LlamaAttention(self.knowledge_config)
        self.knowledge_crossattention = LlamaAttention(self.knowledge_config)
        self.knowledge_intermediate = LlamaDecoderLayer(self.knowledge_config)
        self.knowledge_output = LlamaMLP(self.knowledge_config)

        self.image2knowledge = nn.Linear(config.hidden_size, 256)
        self.knowledge2text = nn.Linear(256, config.hidden_size)
        self.knowledge_fusion_output = LlamaMLP(config)
        # self.knowledge_fusion_output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask=None,
            encoder_attention_mask=None,
            output_attentions=False,
            knowledge_hidden_states=None,
            knowledge_attention_mask=None,
            knowledge_position_matrix=None,
    ):
        # == Begin: Text Self-Attention ==
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # == End  : Text Self-Attention ==

        # == Begin: Knowledge Self-Attention ==
        knowledge_attention_output = self.knowledge_attention(
            knowledge_hidden_states,
            knowledge_attention_mask,
            head_mask=None,
            output_attentions=False,
            past_key_value=None, )[0]
        # == End  : Knowledge Self-Attention ==

        # == Begin: Knowledge-Image Cross-Attention ==
        knowledge_attention_outputs = self.knowledge_crossattention(
            knowledge_attention_output,
            knowledge_attention_mask,
            None,
            self.image2knowledge(encoder_hidden_states),
            encoder_attention_mask,
            None,
            output_attentions,
        )
        knowledge_attention_output = knowledge_attention_outputs[0]
        # == End  : Knowledge-Image Cross-Attention ==

        # == Begin: Text-Image Cross-Attention ==
        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]
        # == End  : Text-Image Cross-Attention ==

        # == Begin: Knowledge-Text Fusion ==
        knowledge = self.knowledge2text(knowledge_attention_output)
        knowledge = torch.bmm(knowledge_position_matrix.to(knowledge_attention_output).permute(0, 2, 1), knowledge)
        attention_output = self.knowledge_fusion_output(attention_output + knowledge, attention_output)
        # == End  : Knowledge-Text Fusion ==

        # == Begin: Text FeedForward ==
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        # == End: Text FeedForward ==

        # == Begin: Knowledge FeedForward ==
        knowledge_layer_output = apply_chunking_to_forward(
            self.knowledge_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim,
            knowledge_attention_output
        )
        knowledge_outputs = (knowledge_layer_output,) + knowledge_attention_outputs[1:]
        # == Begin: Knowledge FeedForward ==

        return outputs, knowledge_outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def knowledge_feed_forward_chunk(self, knowledge_attention_output):
        intermediate_output = self.knowledge_intermediate(knowledge_attention_output)
        layer_output = self.knowledge_output(intermediate_output, knowledge_attention_output)
        return layer_output
