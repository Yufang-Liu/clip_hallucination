# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import argparse

import torch
from clip import load
from clip.model import CLIP
from transformers import CLIPConfig, CLIPModel


def copy_attn_layer(pt_attn_layer, hf_attn_layer):
    pt_attn_layer.in_proj_weight.data = (
        torch.cat([hf_attn_layer.q_proj.weight.data,
                  hf_attn_layer.k_proj.weight.data,
                  hf_attn_layer.v_proj.weight.data], dim=0))
    pt_attn_layer.in_proj_bias.data = (
        torch.cat([hf_attn_layer.q_proj.bias.data,
                  hf_attn_layer.k_proj.bias.data,
                  hf_attn_layer.v_proj.bias.data], dim=0))
    pt_attn_layer.out_proj.weight = hf_attn_layer.out_proj.weight
    pt_attn_layer.out_proj.bias = hf_attn_layer.out_proj.bias


def copy_mlp(pt_mlp, hf_mlp):
    copy_linear(pt_mlp.c_fc, hf_mlp.fc1)
    copy_linear(pt_mlp.c_proj, hf_mlp.fc2)


def copy_linear(pt_linear, hf_linear):
    pt_linear.weight = hf_linear.weight
    pt_linear.bias = hf_linear.bias


def copy_layer(pt_layer, hf_layer,):
    # copy layer norms
    copy_linear(pt_layer.ln_1, hf_layer.layer_norm1)
    copy_linear(pt_layer.ln_2, hf_layer.layer_norm2)

    # copy MLP
    copy_mlp(pt_layer.mlp, hf_layer.mlp,)

    # copy attn
    copy_attn_layer(pt_layer.attn, hf_layer.self_attn)


def copy_layers(pt_layers, hf_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(pt_layer, hf_layer)


def copy_encoder(pt_model, hf_encoder):
    # copy  embeds
    pt_model.token_embedding.weight = hf_encoder.embeddings.token_embedding.weight
    pt_model.positional_embedding = hf_encoder.embeddings.position_embedding.weight

    # copy layer norm
    copy_linear(pt_model.ln_final, hf_encoder.final_layer_norm)

    # copy hidden layers
    copy_layers(pt_model.transformer.resblocks, hf_encoder.encoder.layers)


def copy_text_model_and_projection(pt_model, hf_model):
    # copy projection
    pt_model.text_projection.data = hf_model.text_projection.weight.data.T

    # copy text encoder
    copy_encoder(pt_model, hf_model.text_model)


def copy_vison_model_and_projection(pt_model, hf_model):
    # copy projection
    pt_model.visual.proj.data = hf_model.visual_projection.weight.data.T

    # copy layer norms
    copy_linear(pt_model.visual.ln_pre, hf_model.vision_model.pre_layrnorm)
    copy_linear(pt_model.visual.ln_post, hf_model.vision_model.post_layernorm)

    # copy embeds
    pt_model.visual.conv1.weight.data = hf_model.vision_model.embeddings.patch_embedding.weight.data
    pt_model.visual.class_embedding = hf_model.vision_model.embeddings.class_embedding
    pt_model.visual.positional_embedding.data = hf_model.vision_model.embeddings.position_embedding.weight.data

    # copy encoder
    copy_layers(pt_model.visual.transformer.resblocks, hf_model.vision_model.encoder.layers)


@torch.no_grad()
def convert_hf_model_to_clip_model(hf_model, pixel_size):
    cfg = hf_model.config

    pt_model = CLIP(embed_dim=cfg.text_config.hidden_size,
                    # vision
                    image_resolution=cfg.vision_config.image_size,
                    vision_layers=cfg.vision_config.num_hidden_layers,
                    vision_width=cfg.vision_config.hidden_size,
                    vision_patch_size=cfg.vision_config.patch_size,
                    # text
                    context_length=77,
                    vocab_size=49408,
                    transformer_width=cfg.text_config.hidden_size,
                    transformer_heads=cfg.text_config.num_attention_heads,
                    transformer_layers=12).eval()

    copy_text_model_and_projection(pt_model, hf_model)
    copy_vison_model_and_projection(pt_model, hf_model)
    pt_model.logit_scale = hf_model.logit_scale

    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, pixel_size, pixel_size)

    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    hf_logits_per_image = hf_outputs.logits_per_image
    hf_logits_per_text = hf_outputs.logits_per_text
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3)
    return pt_model

@torch.no_grad()
def convert_hf_model_to_clip_checkpoint(hf_model, pixel_size):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    hf_model = CLIPModel.from_pretrained(hf_model).eval()
    cfg = hf_model.config

    pt_model = CLIP(embed_dim=cfg.text_config.hidden_size,
                 # vision
                 image_resolution=cfg.vision_config.image_size,
                 vision_layers=cfg.vision_config.num_hidden_layers,
                 vision_width=cfg.vision_config.hidden_size,
                 vision_patch_size=cfg.vision_config.patch_size,
                 # text
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=cfg.text_config.hidden_size,
                 transformer_heads=cfg.text_config.num_attention_heads,
                 transformer_layers=12).eval()

    copy_text_model_and_projection(pt_model, hf_model)
    copy_vison_model_and_projection(pt_model, hf_model)
    pt_model.logit_scale = hf_model.logit_scale

    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, pixel_size, pixel_size)

    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    hf_logits_per_image = hf_outputs.logits_per_image
    hf_logits_per_text = hf_outputs.logits_per_text
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3)

    #hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path",
                        default='../cache/transferred/clip_patch14.pt', type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--hf_checkpoint_path",
                        default='../cache/model/clip-vit-large-patch14-336/', type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path",
                        default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_hf_model_to_clip_checkpoint(args.hf_checkpoint_path, args.pytorch_dump_folder_path)