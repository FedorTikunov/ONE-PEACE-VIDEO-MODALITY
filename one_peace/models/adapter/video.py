import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout
from transformers import CLIPImageProcessor, CLIPVisionModel
from .components import Embedding, trunc_normal_, LayerNorm

class VideoAdapter(nn.Module):
    def __init__(self, cfg, embed_dim, attention_heads, num_layers=None):
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(cfg.vision_tower_name)
        self.feature_select = lambda x: x.hidden_states[-1]
        self.dropout_module = FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
        self.alpha = cfg.shrink_alpha

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if cfg.add_type_embedding:
            self.type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.type_embedding_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.type_embedding = None
            self.type_embedding_2 = None

        self.bucket_size = cfg.bucket_size
        self.pos_embed = nn.Parameter(torch.zeros(self.bucket_size**2 + 1, embed_dim))
        position_idx = torch.arange(self.bucket_size**2 + 1)
        self.register_buffer("position_idx", position_idx)

        if cfg.use_attn_bias:
            num_rel_dis = (2 * self.bucket_size - 1) * (2 * self.bucket_size - 1) + 3
            rp_bucket = self.make_video_bucket_position(self.bucket_size, num_rel_dis)
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table_list = nn.ModuleList(
                [Embedding(num_rel_dis, attention_heads, zero_init=True) for _ in range(num_layers or 1)]
            )
        else:
            self.rel_pos_table_list = None

        trunc_normal_(self.cls_embedding)
        trunc_normal_(self.pos_embed)

    @staticmethod
    def make_video_bucket_position(bucket_size, num_relative_distance):
        coords_h = torch.arange(bucket_size)
        coords_w = torch.arange(bucket_size)
        coords_t = torch.arange(bucket_size)
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w]))  # 3, T, H, W
        coords_flatten = torch.flatten(coords, 1)  # 3, T*H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, T*H*W, T*H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # T*H*W, T*H*W, 3
        relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += bucket_size - 1
        relative_coords[:, :, 2] += bucket_size - 1
        relative_coords[:, :, 0] *= 2 * bucket_size - 1
        relative_coords[:, :, 1] *= 2 * bucket_size - 1
        relative_position_index = torch.zeros(size=(bucket_size**2 + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # T*H*W, T*H*W
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
        return relative_position_index

    def get_rel_pos_bias(self, bsz):
        rel_pos_bias_list = []
        for rel_pos_table in self.rel_pos_table_list:
            rp_bucket = self.rp_bucket
            values = rel_pos_table(rp_bucket).unsqueeze(0).expand(bsz, -1, -1, -1)
            values = values.permute(0, 3, 1, 2)
            rel_pos_bias_list.append(values)
        return rel_pos_bias_list

    def forward(self, videos):
        if isinstance(videos, list):
            video_features = []
            for video in videos:
                video_forward_out = self.vision_tower(video.unsqueeze(0), output_hidden_states=True)
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
            video_features = torch.cat(video_features, dim=0)  # Combine list to tensor
        else:
            video_forward_outs = self.vision_tower(videos, output_hidden_states=True)
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

        bsz = video_features.size(0)
        window_size = video_features.size(1)
        padding_mask = videos.new_zeros((bsz, window_size + 1)).bool()
        pos_embed = self.pos_embed.unsqueeze(0).expand(bsz, -1, -1)
        if self.rel_pos_table_list is not None:
            self_attn_bias_list = self.get_rel_pos_bias(bsz)
        else:
            self_attn_bias_list = None

        cls_embedding = self.cls_embedding.expand(bsz, -1, -1)
        adapter_embedding = torch.cat([cls_embedding, video_features], dim=1)
        if self.layernorm_embedding is not None:
            adapter_embedding = self.layernorm_embedding(adapter_embedding)
        if self.alpha != 1.0:
            adapter_embedding = adapter_embedding * self.alpha + adapter_embedding.detach() * (1 - self.alpha)

        x = adapter_embedding + pos_embed

        if self.type_embedding is not None:
            x += self.type_embedding.expand_as(x)
        if self.type_embedding_2 is not None:
            x += self.type_embedding_2.expand_as(x)
        x = self.dropout_module(x)

        return x, padding_mask, self_attn_bias_list

    def gather_features(self, adapter_embedding, pos_embed, self_attn_bias_list, position_ids):
        seq_len, embed_dim = adapter_embedding.shape[-2:]
        gather_seq_len = position_ids.size(1)
        adapter_embedding = adapter_embedding.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))
        pos_embed = pos_embed.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))

        if self_attn_bias_list is not None:
            new_self_attn_bias_list = []
            for self_attn_bias in self_attn_bias_list:
                self_attn_bias = self_attn_bias.gather(
                    2, position_ids[:, None, :, None].expand(-1, self_attn_bias.size(1), -1, seq_len)
                ).gather(3, position_ids[:, None, None, :].expand(-1, self_attn_bias.size(1), gather_seq_len, -1))
                new_self_attn_bias_list.append(self_attn_bias)
        else:
            new_self_attn_bias_list = None

        return adapter_embedding, pos_embed, new_self_attn_bias_list

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        if prefix + 'rel_pos_table.weight' in state_dict:
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table.weight']
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = rel_pos_table_weight
            del state_dict[prefix + 'rel_pos_table.weight']

        if self.rel_pos_table_list is not None and len(self.rel_pos_table_list) > 1 and \
           prefix + 'rel_pos_table_list.1.weight' not in state_dict:
            logger.info('copy rel_pos_weight to each layer')
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table_list.0.weight']
            for i in range(len(self.rel_pos_table_list)):
                state_dict[prefix + 'rel_pos_table_list.{}.weight'.format(i)] = rel_pos_table_weight.clone()

        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        return state_dict
