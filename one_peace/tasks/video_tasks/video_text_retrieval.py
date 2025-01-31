from dataclasses import dataclass, field
from typing import Optional
import logging
import json
import torch
import torch.distributed as dist

from fairseq.tasks import register_task
from fairseq.utils import move_to_cuda

from ..base_task import BaseTask, BaseTaskConfig
from ...data.video_data.video_text_retrieval_dataset import VideoTextRetrievalDataset
from ...utils.data_utils import new_islice, all_gather
from ...metrics import Recall

logger = logging.getLogger(__name__)

@dataclass
class VideoTextRetrievalConfig(BaseTaskConfig):
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation file in JSON format."},
    )
    use_template: bool = False
    patch_video_size: int = field(
        default=256,
        metadata={"help": "Patch size for video frames."}
    )

@register_task("video_text_retrieval", dataclass=VideoTextRetrievalConfig)
class VideoTextRetrievalTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Recall()
        self.text_ids = None
        self.texts = None

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        if self.text_ids is None and self.cfg.valid_file is not None:
            self.text_ids = []
            self.texts = []
            with open(self.cfg.valid_file, 'r') as f:
                valid_data = json.load(f)
                for text_id, text_list in valid_data.items():
                    for text in text_list:
                        self.text_ids.append(int(text_id))
                        self.texts.append(text)
            self.text_ids = torch.tensor(self.text_ids).cuda()

        self.datasets[split] = VideoTextRetrievalDataset(
            split=split,
            dataset=dataset,
            bpe=self.bpe,
            dictionary=self.dict,
            max_src_length=self.cfg.max_src_length,
            patch_video_size=self.cfg.patch_video_size
        )

    @torch.no_grad()
    def begin_valid_epoch(self, epoch, model, subset):
        assert self.text_ids is not None and self.texts is not None
        model.eval()

        dataset = self.datasets[subset]
        text_cnt = len(self.text_ids)

        if dist.is_initialized():
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        else:
            slice_id = 0
            slice_count = 1
        batch_sampler = new_islice(range(text_cnt), slice_id, text_cnt, slice_count)
        start_idx = batch_sampler[0]
        end_idx = batch_sampler[-1] + 1

        text_logits_list = []
        for i in range(start_idx, end_idx, 50):
            samples_list = []
            for text in self.texts[i:min(i+50, end_idx)]:
                text_input = "This is a video of " + text if self.cfg.use_template else text
                item_tuple = (0, None, text_input)
                sample = dataset.__getitem__(0, item_tuple)
                samples_list.append(sample)
            samples = dataset.collater(samples_list)
            samples = move_to_cuda(samples)
            src_tokens = samples["net_input"]["src_tokens"]
            text_logits = model(src_tokens=src_tokens, encoder_type='text')
            text_logits_list.append(text_logits)

        text_logits = torch.cat(text_logits_list, dim=0)
        if dist.is_initialized():
            text_logits = all_gather(text_logits)
        self.metric.initialize(self.text_ids, text_logits)

    @torch.no_grad()
    def valid_step(self, sample, model, criterion, is_dummy_batch):
        loss = 0
        sample_size = len(sample['id'])
        logging_output = {'nsentences': sample_size, 'ntokens': sample_size}
        if not is_dummy_batch:
            model.eval()
            self.eval_step(model, sample)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def eval_step(self, model, sample):
        src_videos = sample["net_input"]["src_videos"]
        video_ids = torch.tensor(sample['id']).to(src_videos.device)
        video_logits = model(src_videos=src_videos, encoder_type='video')
        self.metric.compute(video_ids, video_logits)

    @torch.no_grad()
    def merge_results(self, output_predict=False):
        stats = self.metric.merge_results(output_predict=output_predict)
        for key in list(stats.keys()):
            if key.startswith('img'):
                stats[key.replace('img', 'video')] = stats[key]
                del stats[key]
        return stats
