import math
import torch
from PIL import Image
import decord

from ..base_dataset import BaseDataset
from ...utils.data_utils import get_whole_word_mask, compute_block_mask_1d

import math
import torch
from PIL import Image

class VideoTextPretrainDataset(BaseDataset):
    def __init__(self, split, dataset, bpe, dictionary, 
                 max_src_length=70, num_frames=16, 
                 video_mask_ratio=0.75, text_mask_ratio=0.4, num_patches = 576):
        super().__init__(split, dataset, bpe, dictionary)
        self.num_frames = num_frames
        self.video_mask_ratio = video_mask_ratio
        self.text_mask_ratio = text_mask_ratio
        self.mask_whole_word = get_whole_word_mask(bpe, dictionary)
        self.max_src_length = max_src_length
        self.num_patches = num_patches

    def __getitem__(self, index, item_tuple=None):
        # Dataset is assumed to be a list of dicts loaded from JSON.
        if item_tuple is None:
            item = self.dataset[index]
        else:
            item = {
                "id": item_tuple[0],
                "mp4_path": item_tuple[1],
                "description": item_tuple[2],
            }
        video_path = item['mp4_path']
        text = item['description']
        
        # Process video using OpenCV via process_video()
        frames = self.read_video(video_path, self.num_frames)
        
        # Process text: assume process_text, encode_text, add_whole_word_mask are inherited from BaseDataset.
        processed_text = self.process_text(text)
        text_item = self.encode_text(f' {processed_text}', self.max_src_length, append_eos=False)
        vid_text_mask_indices= self.add_whole_word_mask(text_item, self.text_mask_ratio)
        text_item = torch.cat([text_item, torch.LongTensor([self.eos])])
        vid_text_preserve_ids = (~vid_text_mask_indices).nonzero(as_tuple=True)[0]
        
        frame_mask = torch.rand(self.num_frames) < self.video_mask_ratio
        patch_mask = frame_mask.unsqueeze(1).repeat(1, self.num_patches).view(-1)
        vid_video_mask_indices = torch.cat([torch.BoolTensor([False]).to(patch_mask.device), patch_mask])

        frame_mask = torch.rand(self.num_frames) < self.video_mask_ratio
        patch_mask = frame_mask.unsqueeze(1).repeat(1, self.num_patches).view(-1)
        video_mask_indices = torch.cat([torch.BoolTensor([False]).to(patch_mask.device), patch_mask])

        video_mask_indices = torch.cat([torch.BoolTensor([False]), video_mask_indices])
        video_preserve_ids = (~video_mask_indices).nonzero(as_tuple=True)[0]

        vid_video_preserve_ids = (~vid_video_mask_indices).nonzero(as_tuple=True)[0]
        return {
            "id": item["id"],
            "source_video": frames,       # List of PIL Images
            "source_text": text_item,     # Tokenized text tensor
            "video_mask_indices": video_mask_indices,
            "video_preserve_ids": video_preserve_ids,
            "vid_text_mask_indices": vid_text_mask_indices,
            "vid_text_preserve_ids": vid_text_preserve_ids,
            "vid_video_mask_indices": vid_video_mask_indices,
            "vid_video_preserve_ids": vid_video_preserve_ids,
        }

    def sample_frames(self, total_frames):
        # Sample self.num_frames evenly spaced frames.
        step = max(total_frames // self.num_frames, 1)
        return [i * step for i in range(self.num_frames) if i * step < total_frames]


    def add_whole_word_mask(self, source, p):
        is_word_start = self.mask_whole_word.gather(0, source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        assert num_to_mask != 0

        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_indices = torch.zeros(len(source)).bool()
        mask_indices[indices] = True

        is_word_start = torch.cat([is_word_start, torch.Tensor([255]).type_as(is_word_start)])
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_indices[indices] = True

        return mask_indices