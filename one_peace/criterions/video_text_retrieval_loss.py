from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


def adjust_label_smoothed_nll_loss(lprobs, target, epsilon=0.0):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if epsilon != 0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    else:
        loss = nll_loss
    return loss.mean()


@torch.no_grad()
def gather_without_grad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@dataclass
class VideoTextRetrievalCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("video_text_retrieval_criterion", dataclass=VideoTextRetrievalCriterionConfig)
class VideoTextRetrievalCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing=0.0):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        src_tokens = sample['net_input']['src_tokens']
        src_videos = sample['net_input']['src_videos']
        video_padding_masks = sample['net_input']['video_padding_masks']

        text_logits = model(src_tokens=src_tokens, encoder_type='text')
        video_logits = model(
            src_videos=src_videos,
            video_padding_masks=video_padding_masks,
            encoder_type='video'
        )
        text_logits_all = gather_without_grad(text_logits) if dist.is_initialized() else text_logits.data
        video_logits_all = gather_without_grad(video_logits) if dist.is_initialized() else video_logits.data

        logit_scale_exp = model(return_logit_scale=True)

        # compute vtc loss
        vtc_loss, v2t_ncorrect, t2v_ncorrect = self.compute_vtc_loss(
            video_logits, text_logits,
            video_logits_all, text_logits_all,
            logit_scale_exp
        )

        sample_size = 1
        logging_output = {
            "loss": vtc_loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
            "v2t_ncorrect": v2t_ncorrect,
            "t2v_ncorrect": t2v_ncorrect,
            "logit_scale_exp": logit_scale_exp
        }
        return vtc_loss, sample_size, logging_output

    def compute_vtc_loss(self, video_logits, text_logits, video_logits_all, text_logits_all, logit_scale_exp):
        slice_id = dist.get_rank() if dist.is_initialized() else 0
        bsz = video_logits.size(0)
        start_idx = bsz * slice_id
        end_idx = start_idx + bsz
        targets = torch.arange(start_idx, end_idx).to(video_logits.device)

        sim_v2t = logit_scale_exp * video_logits @ text_logits_all.t()
        sim_t2v = logit_scale_exp * text_logits @ video_logits_all.t()
        log_sim_v2t = utils.log_softmax(sim_v2t, dim=-1).type_as(sim_v2t)
        log_sim_t2v = utils.log_softmax(sim_t2v, dim=-1).type_as(sim_t2v)
        v2t_loss = adjust_label_smoothed_nll_loss(log_sim_v2t, targets, self.label_smoothing)
        t2v_loss = adjust_label_smoothed_nll_loss(log_sim_t2v, targets, self.label_smoothing)
        vtc_loss = (v2t_loss + t2v_loss) / 2

        with torch.no_grad():
            v2t_preds = sim_v2t.argmax(dim=1)
            t2v_preds = sim_t2v.argmax(dim=1)
            v2t_ncorrect = (v2t_preds == targets).float().sum()
            t2v_ncorrect = (t2v_preds == targets).float().sum()

        return vtc_loss, v2t_ncorrect, t2v_ncorrect

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        logit_scale_exp_sum = sum(log.get("logit_scale_exp", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "logit_scale_exp", logit_scale_exp_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

        if len(logging_outputs) > 0 and "v2t_ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("v2t_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "v2t_accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "t2v_ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("t2v_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "t2v_accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
