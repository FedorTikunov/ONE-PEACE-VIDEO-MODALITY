import math
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
from ...utils.data_utils import get_whole_word_mask

class VideoTextPretrainDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_video_size=256,
        text_mask_ratio=0.15,
        video_mask_ratio=0.75,
        vl_text_mask_ratio=0.4,
        vl_video_mask_ratio=0.6875,
        min_scale=0.9
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.mask_whole_word = (get_whole_word_mask(bpe, dictionary))

        self.text_mask_ratio = text_mask_ratio
        self.video_mask_ratio = video_mask_ratio
        self.vl_text_mask_ratio = vl_text_mask_ratio
        self.vl_video_mask_ratio = vl_video_mask_ratio

        self.num_patches = (patch_video_size // 16) ** 2
        self.patch_video_size = patch_video_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    patch_video_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((patch_video_size, patch_video_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        uniq_id, video, caption = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or uniq_id.isdigit() else uniq_id

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length, append_eos=False)
        text_mask_ratio = self.text_mask_ratio
        text_mask_indices = self.add_whole_word_mask(text_src_item, text_mask_ratio)

        vl_text_mask_ratio = self.vl_text_mask_ratio
        vl_text_mask_len = int(len(text_mask_indices) * vl_text_mask_ratio)
        vl_text_mask_ids = torch.randn(*text_mask_indices.size()).masked_fill(
            text_mask_indices, -float('inf')
        ).argsort(descending=True)[:vl_text_mask_len]
        vl_text_mask_indices = torch.zeros(len(text_mask_indices)).scatter(0, vl_text_mask_ids, 1).bool()

        if video is not None:
            video_frames = self.read_video(video)
            patch_video = self.transform(video_frames)
        else:
            patch_video = torch.randn((self.patch_video_size, self.patch_video_size))

        mask_patches = int(self.num_patches * self.video_mask_ratio)
        random_ids = torch.randperm(self.num_patches)[:mask_patches]
        video_mask_indices = torch.zeros(self.num_patches).scatter(0, random_ids, 1).bool()

        vl_mask_patches = int(self.num_patches * self.vl_video_mask_ratio)
        vl_video_mask_ids = torch.randn(*video_mask_indices.size()).masked_fill(
            ~video_mask_indices, -float('inf')
        ).argsort(descending=True)[:(vl_mask_patches - (self.num_patches - mask_patches))]
        vl_video_mask_ids = torch.cat([vl_video_mask_ids, (~video_mask_indices).nonzero(as_tuple=True)[0]])
        vl_video_mask_indices = torch.zeros(self.num_patches).scatter(0, vl_video_mask_ids, 1).bool()

        text_src_item = torch.cat([text_src_item, torch.LongTensor([self.eos])])
        text_mask_indices = torch.cat([torch.BoolTensor([False]), text_mask_indices, torch.BoolTensor([False])])
        text_preserve_ids = (~text_mask_indices).nonzero(as_tuple=True)[0]
        video_mask_indices = torch.cat([torch.BoolTensor([False]), video_mask_indices])
        video_preserve_ids = (~video_mask_indices).nonzero(as_tuple=True)[0]
        vl_text_mask_indices = torch.cat([torch.BoolTensor([False]), vl_text_mask_indices, torch.BoolTensor([False])])
        vl_text_preserve_ids = (~vl_text_mask_indices).nonzero(as_tuple=True)[0]
        vl_video_mask_indices = torch.cat([torch.BoolTensor([False]), vl_video_mask_indices])
        vl_video_preserve_ids = (~vl_video_mask_indices).nonzero(as_tuple=True)[0]

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "text_mask_indices": text_mask_indices,
            "text_preserve_ids": text_preserve_ids,
            "source_video": patch_video,
            "video_mask_indices": video_mask_indices,
            "video_preserve_ids": video_preserve_ids,
            "vl_text_mask_indices": vl_text_mask_indices,
            "vl_text_preserve_ids": vl_text_preserve_ids,
            "vl_video_mask_indices": vl_video_mask_indices,
            "vl_video_preserve_ids": vl_video_preserve_ids,
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
