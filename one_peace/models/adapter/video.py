# video.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel
from fairseq.modules import FairseqDropout
import logging

logger = logging.getLogger(__name__)

from ..components import Embedding, trunc_normal_, LayerNorm

def make_video_bucket_position(bucket_size, num_frames, num_relative_distance):
    """
    Creates a bucket position tensor for relative positional embedding in 3D space (time, height, width).
    """
    coords_t = torch.arange(num_frames)
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))  # [3, T, H, W]
    coords_flatten = torch.flatten(coords, 1)  # [3, T*H*W]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]

    # Shift to start from 0
    relative_coords[:, :, 0] += num_frames - 1
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 2] += bucket_size - 1

    relative_coords[:, :, 0] *= (2 * bucket_size - 1) * (2 * bucket_size - 1)
    relative_coords[:, :, 1] *= (2 * bucket_size - 1)

    relative_position_index = relative_coords.sum(-1)  # [N, N]

    num_relative_distance = (2 * num_frames - 1) * (2 * bucket_size - 1) ** 2 + 3
    rp_bucket = torch.zeros((relative_coords.size(0) + 1, relative_coords.size(1) + 1), dtype=torch.long)
    rp_bucket[1:, 1:] = relative_position_index + 1
    rp_bucket[0, 0:] = num_relative_distance - 3
    rp_bucket[0:, 0] = num_relative_distance - 2
    rp_bucket[0, 0] = num_relative_distance - 1

    return rp_bucket

class VideoAdapter(nn.Module):
    def __init__(self, cfg, embed_dim, attention_heads, num_layers=None):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_layers = num_layers if num_layers is not None else 1
        self.num_frames = cfg.num_frames
        self.bucket_size = cfg.bucket_size
        # Load CLIP components
        self.clip_model = CLIPVisionModel.from_pretrained(cfg.clip_model_name)
        self.clip_model.eval()  # Set CLIP model to evaluation mode
        #print("DEBUG: Loaded CLIPVisionModel.")

        # Projection layer to match embed_dim if necessary
        if self.clip_model.config.hidden_size != embed_dim:
            self.proj = nn.Linear(self.clip_model.config.hidden_size, embed_dim)
        else:
            self.proj = nn.Identity()
        #print("DEBUG: Projection layer type:", type(self.proj).__name__)

        # Image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.clip_model_name)
        #print("DEBUG: Loaded CLIPImageProcessor.")

        # Positional embeddings: shape [1, total_patches+1, embed_dim]
        total_patches = cfg.num_frames * (cfg.bucket_size ** 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed)
        #print("DEBUG: Initialized pos_embed with shape:", self.pos_embed.shape)

        # CLS token
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_embedding)
        #print("DEBUG: Initialized cls_embedding with shape:", self.cls_embedding.shape)

        # Type embeddings
        if cfg.add_type_embedding:
            self.type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.type_embedding)
            #print("DEBUG: Initialized type_embedding with shape:", self.type_embedding.shape)
        else:
            self.type_embedding = None

        # Layer normalization
        self.layernorm_embedding = nn.LayerNorm(embed_dim) if cfg.layernorm_embedding else None
        #if self.layernorm_embedding is not None:
            #print("DEBUG: Using LayerNorm for embedding with embed_dim:", embed_dim)

        # Relative position bias
        if cfg.use_attn_bias:
            num_rel_dis = (2 * cfg.num_frames - 1) * (2 * cfg.bucket_size - 1) ** 2 + 3
            rp_bucket = make_video_bucket_position(cfg.bucket_size, cfg.num_frames, num_rel_dis)
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table_list = nn.ModuleList([
                nn.Embedding(num_rel_dis, attention_heads)
                for _ in range(self.num_layers)
            ])
            #print("DEBUG: Initialized relative position bias with rp_bucket shape:", rp_bucket.shape)
        else:
            self.rel_pos_table_list = None

        # Dropout
        self.dropout_module = FairseqDropout(cfg.dropout)
        #print("DEBUG: Dropout rate set to:", cfg.dropout)

    def forward(self, frames_batch, preserve_ids=None, preserve_embed=None, mask_token=None, is_second_video=False):
        """
        Args:
            frames_batch: A list of lists of PIL Images (batch_size x num_frames)
        Returns:
            x: Embeddings of shape (batch_size, seq_len, embed_dim)
            padding_mask: Padding mask (bool tensor)
            self_attn_bias_list: List of attention biases (or None)
        """
        device = next(self.parameters()).device
        batch_size = len(frames_batch)
        #print("DEBUG: Batch size =", batch_size)
        all_pixel_values = []

        # Step 1: Preprocess frames using image_processor.
        for i, frames in enumerate(frames_batch):
            inputs = self.image_processor(images=frames, return_tensors='pt')
            pixel_values = inputs['pixel_values']  # [num_frames, 3, H, W]
            all_pixel_values.append(pixel_values)
            #print(f"DEBUG: Processed video {i}: pixel_values shape =", pixel_values.shape)

        # Stack videos: [B, num_frames, 3, H, W]
        pixel_values = torch.stack(all_pixel_values, dim=0).to(device)
        #print("DEBUG: Stacked pixel_values shape =", pixel_values.shape)

        batch_size, num_frames, channels, height, width = pixel_values.size()

        # Step 2: Extract frame embeddings.
        frame_embeddings = self.extract_frame_embeddings(pixel_values)
        #print("DEBUG: Frame embeddings shape =", frame_embeddings.shape)

        # Step 3: Flatten temporal and spatial dimensions.
        # frame_embeddings: [B, num_frames, num_patches, embed_dim]
        batch_size, num_frames, num_patches, _ = frame_embeddings.size()
        adapter_embedding = frame_embeddings.view(batch_size, num_frames * num_patches, self.embed_dim)
        #print("DEBUG: adapter_embedding shape after flattening =", adapter_embedding.shape)

        # Step 4: Add CLS token.
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        adapter_embedding = torch.cat([cls_embedding, adapter_embedding], dim=1)  # [B, N+1, embed_dim]
        #print("DEBUG: adapter_embedding shape after adding CLS token =", adapter_embedding.shape)

        # Step 5: Positional embeddings.
        current_seq_len = adapter_embedding.size(1)
        pos_embed = self.pos_embed.expand(batch_size, -1, -1)
        #print("DEBUG: Original pos_embed shape =", self.pos_embed.shape, "Expanded pos_embed shape =", pos_embed.shape)
        if pos_embed.size(1) != current_seq_len:
            #print("DEBUG: Interpolating pos_embed from", pos_embed.size(1), "to", current_seq_len)
            pos_embed = pos_embed.transpose(1, 2)
            pos_embed = F.interpolate(pos_embed, size=current_seq_len, mode='linear', align_corners=False)
            pos_embed = pos_embed.transpose(1, 2)
            #print("DEBUG: pos_embed shape after interpolation =", pos_embed.shape)
        # Step 6: Add positional embeddings.
        x = adapter_embedding + pos_embed
        #print("DEBUG: x shape after adding pos_embed =", x.shape)

        # Step 7: Optional type embedding.
        if self.type_embedding is not None:
            x += self.type_embedding.expand_as(x)
            #print("DEBUG: x shape after adding type_embedding =", x.shape)
        if is_second_video and hasattr(self, 'type_embedding_2'):
            x += self.type_embedding_2.expand_as(x)
            #print("DEBUG: x shape after adding type_embedding_2 =", x.shape)

        # Step 8: Apply layer normalization and dropout.
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
            #print("DEBUG: x shape after LayerNorm =", x.shape)
        if self.dropout_module.p > 0:
            x = self.dropout_module(x)
            #print("DEBUG: x shape after dropout =", x.shape)

        # Step 9: Compute relative position bias.
        if self.rel_pos_table_list is not None:
            self_attn_bias_list = self.get_rel_pos_bias(batch_size, x.size(1))
            #print("DEBUG: Computed self_attn_bias_list with", len(self_attn_bias_list), "elements")
        else:
            self_attn_bias_list = None

        # Step 10: Create padding mask.
        padding_mask = x.new_zeros((batch_size, x.size(1)), dtype=torch.bool)
        #print("DEBUG: padding_mask shape =", padding_mask.shape)

        return x, padding_mask, self_attn_bias_list

    def extract_frame_embeddings(self, pixel_values):
        """
        Extracts frame embeddings using CLIPVisionModel and projects them to embed_dim.
        """
        batch_size, num_frames, channels, height, width = pixel_values.size()
        pixel_values = pixel_values.view(-1, channels, height, width)  # [B * num_frames, 3, H, W]
        #print("DEBUG: In extract_frame_embeddings: reshaped pixel_values =", pixel_values.shape)
        with torch.no_grad():
            outputs = self.clip_model(pixel_values)
            # Exclude CLS token; outputs.last_hidden_state shape: [B * num_frames, num_patches+1, hidden_size]
            frame_embeddings = outputs.last_hidden_state[:, 1:, :]
        #print("DEBUG: Raw frame_embeddings shape =", frame_embeddings.shape)
        frame_embeddings = self.proj(frame_embeddings)
        #print("DEBUG: Projected frame_embeddings shape =", frame_embeddings.shape)
        num_patches = frame_embeddings.size(1)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, num_patches, self.embed_dim)
        #print("DEBUG: Final frame_embeddings shape =", frame_embeddings.shape)
        return frame_embeddings # [B, num_frames, num_patches, embed_dim]


    def get_rel_pos_bias(self, bsz, seq_len):
        """
        Computes relative position bias.
        """
        rel_pos_bias_list = []
        rp_bucket = self.rp_bucket[:seq_len, :seq_len].to(self.pos_embed.device)
        for rel_pos_table in self.rel_pos_table_list:
            values = rel_pos_table(rp_bucket).permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
            values = values.expand(bsz, -1, -1, -1)  # [bsz, num_heads, seq_len, seq_len]
            rel_pos_bias_list.append(values)
        return rel_pos_bias_list

    def get_embed_positions(self, batch_size):
        """
        Retrieves positional embeddings.
        """
        pos_embed = self.pos_embed.expand(batch_size, -1, -1)  # [B, total_patches + 1, embed_dim]
        return pos_embed

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Upgrades the model's state dict to be compatible with the current model configuration.
        """
        prefix = name + "." if name != "" else ""

        # Handle relative position bias
        if prefix + 'rel_pos_table.weight' in state_dict:
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table.weight']
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = rel_pos_table_weight
            del state_dict[prefix + 'rel_pos_table.weight']

        # Interpolate relative position bias if necessary
        if prefix + 'rel_pos_table_list.0.weight' in state_dict and \
            (2 * self.num_frames - 1) * (2 * self.bucket_size - 1) ** 2 + 3 > state_dict[prefix + 'rel_pos_table_list.0.weight'].size(0):
            logger.info('Interpolate relative position embedding for VideoAdapter')
            num_extra_tokens = 3
            num_attn_heads = state_dict[prefix + 'rel_pos_table_list.0.weight'].size(-1)
            src_size = int(round(((state_dict[prefix + 'rel_pos_table_list.0.weight'].size(0) - num_extra_tokens) ** (1/3))))
            dst_size_t = 2 * self.num_frames - 1
            dst_size_s = 2 * self.bucket_size - 1

            extra_tokens = state_dict[prefix + 'rel_pos_table_list.0.weight'][-num_extra_tokens:, :]
            rel_pos_bias = state_dict[prefix + 'rel_pos_table_list.0.weight'][:-num_extra_tokens, :]

            # Reshape to 3D grid
            rel_pos_bias = rel_pos_bias.view(src_size, src_size, src_size, num_attn_heads)
            # Interpolate
            new_rel_pos_bias = F.interpolate(
                rel_pos_bias.permute(3, 0, 1, 2),  # [num_attn_heads, src_size, src_size, src_size]
                size=(dst_size_t, dst_size_s, dst_size_s),
                mode='trilinear',
                align_corners=False
            ).permute(1, 2, 3, 0).contiguous().view(-1, num_attn_heads)
            new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = new_rel_pos_bias
            state_dict[prefix + 'rp_bucket'] = self.rp_bucket

        # Copy rel_pos_weight to each layer if missing
        if self.rel_pos_table_list is not None and len(self.rel_pos_table_list) > 1 \
                and prefix + 'rel_pos_table_list.1.weight' not in state_dict:
            logger.info('Copy rel_pos_weight to each layer in VideoAdapter')
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table_list.0.weight']
            for i in range(len(self.rel_pos_table_list)):
                state_dict[f'{prefix}rel_pos_table_list.{i}.weight'] = rel_pos_table_weight.clone()

        # Interpolate positional embeddings if necessary
        if prefix + 'pos_embed' in state_dict and \
                self.pos_embed.size(1) > state_dict[prefix + 'pos_embed'].size(1):
            logger.info('Interpolate absolute position embedding for VideoAdapter')
            cls_pos_embed = state_dict[prefix + 'pos_embed'][:, :1, :]  # [1, 1, D]
            old_pos_embed = state_dict[prefix + 'pos_embed'][:, 1:, :]  # [1, old_num_patches, D]
            old_num_frames = self.num_frames  # May need to handle this dynamically
            old_bucket_size = int((old_pos_embed.size(1) // old_num_frames) ** 0.5)
            old_pos_embed = old_pos_embed.view(1, old_num_frames, old_bucket_size, old_bucket_size, -1).permute(0, 4, 1, 2, 3)
            # Interpolate
            new_pos_embed = F.interpolate(
                old_pos_embed,
                size=(self.num_frames, self.bucket_size, self.bucket_size),
                mode='trilinear',
                align_corners=False
            ).permute(0, 2, 3, 4, 1).reshape(1, -1, self.embed_dim)
            pos_embed = torch.cat([cls_pos_embed, new_pos_embed], dim=1)
            state_dict[prefix + 'pos_embed'] = pos_embed

        # Initialize missing parameters
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info(f'{prefix + param_name} not found in state_dict. Re-initializing.')
                state_dict[prefix + param_name] = param_tensor.clone()

        return state_dict
