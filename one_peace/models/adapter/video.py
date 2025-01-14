import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from fairseq.modules import FairseqDropout, LayerNorm


class VideoAdapter(nn.Module):

    def __init__(self, vision_tower_name, select_layer, select_feature='patch', cfg=None):
        super(VideoAdapter, self).__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.alpha = cfg.shrink_alpha if cfg else 1.0

        if cfg and cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(self.vision_tower.config.hidden_size)
        else:
            self.layernorm_embedding = None

        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, self.vision_tower.config.hidden_size))
        nn.init.trunc_normal_(self.cls_embedding)

        self.dropout_module = FairseqDropout(cfg.dropout) if cfg else FairseqDropout(0.1)

        # Initialize positional embedding
        self.bucket_size = cfg.bucket_size if cfg else 16
        self.pos_embed = nn.Parameter(torch.zeros(self.bucket_size ** 2 + 1, self.vision_tower.config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed)

    def get_rel_pos_bias(self, bsz, seq_len):
        rel_pos_bias_list = []
        for rel_pos_table in self.rel_pos_table_list:
            rp_bucket = self.rp_bucket[:seq_len, :seq_len]
            values = rel_pos_table(rp_bucket).unsqueeze(0).expand(bsz, -1, -1, -1)
            values = values.permute(0, 3, 1, 2)
            rel_pos_bias_list.append(values)
        return rel_pos_bias_list

    def gather_features(self, adapter_embedding, pos_embed, self_attn_bias_list, position_ids):
        seq_len, embed_dim = adapter_embedding.shape[-2:]
        gather_seq_len = position_ids.size(1)
        adapter_embedding = adapter_embedding.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))
        pos_embed = pos_embed.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))

        if self_attn_bias_list is not None:
            new_self_attn_bias_list = []
            for self_attn_bias in self_attn_bias_list:
                self_attn_bias = self_attn_bias.gather(2, position_ids[:, None, :, None].expand(-1, self_attn_bias.size(1), -1, seq_len))
                self_attn_bias = self_attn_bias.gather(3, position_ids[:, None, None, :].expand(-1, self_attn_bias.size(1), gather_seq_len, -1))
                new_self_attn_bias_list.append(self_attn_bias)
            return adapter_embedding, pos_embed, new_self_attn_bias_list

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, video_frames, preserve_ids=None, preserve_embed=None, mask_token=None, is_second_image=False):
        bsz = len(video_frames)
        window_size = video_frames[0].size(1) // 16

        padding_mask = video_frames[0].new_zeros((bsz, window_size ** 2 + 1)).bool()
        pos_embed = self.get_embed_positions(bsz, window_size)

        if self.rel_pos_table_list is not None:
            self_attn_bias_list = self.get_rel_pos_bias(bsz, seq_len=window_size**2+1)
        else:
            self_attn_bias_list = None

        if preserve_embed is not None:
            seq_len, embed_dim = pos_embed.size(1), pos_embed.size(2)
            adapter_embedding = mask_token.repeat(bsz * seq_len, 1)
            right_preserve_indices = torch.nonzero(preserve_ids.ne(-1).flatten(), as_tuple=False).flatten()
            left_preserve_indices = preserve_ids + (torch.arange(bsz) * seq_len).unsqueeze(1).type_as(preserve_ids)
            left_preserve_indices = left_preserve_indices.view(-1)[right_preserve_indices]
            adapter_embedding[left_preserve_indices] = preserve_embed.reshape(-1, embed_dim)[right_preserve_indices]
            adapter_embedding = adapter_embedding.reshape(bsz, seq_len, embed_dim)
        else:
            processed_frames = [self.image_processor(frame, return_tensors='pt')['pixel_values'].squeeze(0) for frame in video_frames]
            frame_batch = torch.stack(processed_frames).to(self.vision_tower.device)
            image_forward_outs = self.vision_tower(frame_batch, output_hidden_states=True)
            adapter_embedding = self.feature_select(image_forward_outs).flatten(2).transpose(1, 2)
            cls_embedding = self.cls_embedding.expand(bsz, -1, -1)
            adapter_embedding = torch.cat([cls_embedding, adapter_embedding], dim=1)

            if preserve_ids is not None:
                padding_mask = preserve_ids.eq(-1)
                position_ids = preserve_ids.masked_fill(preserve_ids.eq(-1), preserve_ids.size(1) - 1)
                adapter_embedding, pos_embed, self_attn_bias_list = self.gather_features(adapter_embedding, pos_embed, self_attn_bias_list, position_ids)

            if self.layernorm_embedding is not None:
                adapter_embedding = self.layernorm_embedding(adapter_embedding)

            if self.alpha != 1.0:
                adapter_embedding = adapter_embedding * self.alpha + adapter_embedding.detach() * (1 - self.alpha)

        x = adapter_embedding + pos_embed

        if self.type_embedding is not None:
            x += self.type_embedding.expand_as(x)
        if is_second_image and self.type_embedding_2 is not None:
            x += self.type_embedding_2.expand_as(x)
        x = self.dropout_module(x)

        return x, padding_mask, self_attn_bias_list
