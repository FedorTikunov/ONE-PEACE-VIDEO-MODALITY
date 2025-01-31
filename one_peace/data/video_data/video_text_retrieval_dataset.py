import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ..base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD

class VideoTextRetrievalDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_video_size=256
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_video_size = patch_video_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    patch_video_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC
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
        item_tuple = self.dataset[index] if item_tuple is none else item_tuple
        uniq_id, video, caption = item_tuple
        if uniq_id is not None:
            uniq_id = int(uniq_id) if isinstance(uniq_id, int) or uniq_id.isdigit() else uniq_id

        if video is not None:
            video_frames = self.read_video(video)
            patch_video = self.transform(video_frames)
        else:
            patch_video = torch.randn((3, self.patch_video_size, self.patch_video_size))

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_video": patch_video,
        }
        return example
