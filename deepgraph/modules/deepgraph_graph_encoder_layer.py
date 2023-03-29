# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .multihead_attention import MultiheadAttention,MultiheadAttentionRe
import math

class DeepGraphGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
        deepnorm: bool = False,
        encoder_layers: int = 12,
        reattention:bool=False
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            reattention=reattention
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)


        if deepnorm:
            # self.shortcut_scale = math.pow(math.pow(cfg.encoder_layers, 4) * cfg.decoder_layers, 0.0625) * 0.81
            # if utils.safe_getattr(cfg, "deepnorm_encoder_only", False):
            self.shortcut_scale = math.pow(2.0 * encoder_layers, 0.25)
            # print('deepnorm shortcut_scale ',self.shortcut_scale)
        else:
            self.shortcut_scale = 1.0

        if deepnorm:
            # self.fixup_scale = math.pow(math.pow(cfg.encoder_layers, 4) * cfg.decoder_layers, 0.0625) / 1.15
            # if utils.safe_getattr(cfg, "deepnorm_encoder_only", False):
            self.fixup_scale = math.pow(8.0 * encoder_layers, 0.25)
            # print('deepnorm fixup_scale ', self.fixup_scale)
            self.deepnorm_init()

    def deepnorm_init(self):
        def rescale(param):
            param.div_(self.fixup_scale)

        rescale(self.self_attn.v_proj.weight.data)
        rescale(self.self_attn.v_proj.bias.data)
        rescale(self.self_attn.out_proj.weight.data)
        rescale(self.self_attn.out_proj.bias.data)

        rescale(self.fc1.weight.data)
        rescale(self.fc2.weight.data)
        rescale(self.fc1.bias.data)
        rescale(self.fc2.bias.data)
        # print('deepnorm init')

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        reattention=False
    ):
        if not reattention:
            return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
        else:
            return MultiheadAttentionRe(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )


    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual* self.shortcut_scale + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual * self.shortcut_scale + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn
