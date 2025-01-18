import math
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..base_dataset import BaseDataset
from ...utils.data_utils import get_whole_word_mask, compute_block_mask_1d

class VideoTextRetrievalDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        text_mask_ratio=0.15,
        vl_text_mask_ratio=0.4,
        video_frame_size=224,  # Typical size used by CLIP
        min_scale=0.9
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.mask_whole_word = get_whole_word_mask(bpe, dictionary)

        self.text_mask_ratio = text_mask_ratio
        self.vl_text_mask_ratio = vl_text_mask_ratio

        self.video_frame_size = video_frame_size
        self.min_scale = min_scale

        mean = [0.485, 0.456, 0.406]  # Typical mean for normalization
        std = [0.229, 0.224, 0.225]  # Typical std for normalization

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    video_frame_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((video_frame_size, video_frame_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        uniq_id, video_frames, caption = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or uniq_id.isdigit() else uniq_id

        # Process text
        caption = self.process_text(caption)
        text_src_item = self.encode_text(f' {caption}', self.max_src_length, append_eos=False)
        vl_text_mask_indices = self.add_whole_word_mask(text_src_item, self.vl_text_mask_ratio)
        text_src_item = torch.cat([text_src_item, torch.LongTensor([self.eos])])

        # Process video frames
        processed_frames = [self.transform(frame) for frame in video_frames]

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "text_mask_indices": vl_text_mask_indices,
            "source_video": processed_frames,
        }
        return example

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
