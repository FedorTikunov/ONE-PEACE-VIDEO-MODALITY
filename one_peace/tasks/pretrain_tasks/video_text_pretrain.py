from dataclasses import dataclass, field
from typing import Optional
import logging
import json
import torch

from fairseq.tasks import register_task
from fairseq.utils import move_to_cuda

from ..base_task import BaseTask, BaseTaskConfig
from ...data.pretrain_data.video_text_pretrain_dataset import VideoTextPretrainDataset
from ...metrics import Recall

logger = logging.getLogger(__name__)

@dataclass
class VideoTextPretrainConfig(BaseTaskConfig):
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "Validation file in JSON format."},
    )
    num_frames: int = field(
        default=16,
        metadata={"help": "Number of frames per video."},
    )
    video_mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Masking ratio for video frames."},
    )
    text_mask_ratio: float = field(
        default=0.4,
        metadata={"help": "Masking ratio for text."},
    )

@register_task("video_text_pretrain", dataclass=VideoTextPretrainConfig)
class VideoTextPretrainTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Recall()
        self.text_ids = None
        self.texts = None

    def load_dataset(self, split, epoch=1, **kwargs):
        # Load raw dataset (JSON list) via BaseTask (which now supports JSON)
        dataset = super().load_dataset(split, epoch, **kwargs)
        if self.text_ids is None and self.cfg.valid_file is not None:
            print("self.text_ids:", self.text_ids)
            print("self.texts:", self.texts)
            self.text_ids = []
            self.texts = []
            with open(self.cfg.valid_file, "r") as f:
                valid_data = json.load(f)
                for item in valid_data:
                    self.text_ids.append(item["id"])
                    self.texts.append(item["description"])
            self.text_ids = torch.tensor(self.text_ids).cuda()
        self.datasets[split] = VideoTextPretrainDataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            num_frames=self.cfg.num_frames,
            video_mask_ratio=self.cfg.video_mask_ratio,
            text_mask_ratio=self.cfg.text_mask_ratio,
        )

    @torch.no_grad()
    def begin_valid_epoch(self, epoch, model, subset):
        # During validation, we extract text features for retrieval evaluation.
        assert self.text_ids is not None and self.texts is not None
        model.eval()

        dataset = self.datasets[subset]

        text_logits_list = []
        dummy_samples = []
        for text in self.texts:
            dummy_tuple = (0, "/userspace/tfv/dataset/finevideo/finevideo_segments/sample_34145/scene_1_activity_2.mp4", text)
            sample = dataset.__getitem__(0, dummy_tuple)
            dummy_samples.append(sample)
        samples = dataset.collater(dummy_samples)
        samples = move_to_cuda(samples)
        # After collating, text tokens are merged into net_input["src_tokens"]
        src_tokens = samples["net_input"]["src_tokens"]
        text_logits, _ = model(src_tokens=src_tokens, encoder_type='text')
        text_logits_list.append(text_logits)

        text_logits = torch.cat(text_logits_list, dim=0)
        self.metric.initialize(self.text_ids, text_logits)

    @torch.no_grad()
    def valid_step(self, sample, model, criterion, is_dummy_batch):
        loss = 0
        sample_size = len(sample['id'])
        logging_output = {'nsentences': 1, 'ntokens': 1}
        if not is_dummy_batch:
            model.eval()
            self.eval_step(model, sample)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def eval_step(self, model, sample):
        # Collated video tensors are stored under net_input["src_videos"]
        video_frames = sample["net_input"]["src_videos"]
        #video_ids = torch.tensor(sample['id']).to(video_frames.device)
        video_logits, _ = model(src_videos=video_frames, encoder_type='video')
        video_ids = torch.tensor(sample['id']).to(video_logits.device)
        self.metric.compute(video_ids, video_logits)

    @torch.no_grad()
    def merge_results(self, output_predict=False):
        stats = self.metric.merge_results(output_predict=output_predict)
        # Rename any keys starting with 'img' to 'video'
        for key in list(stats.keys()):
            if key.startswith('img'):
                stats[key.replace('img', 'video')] = stats[key]
                del stats[key]
        return stats
