from dataclasses import dataclass, field
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
class VideoTextPretrainLossConfig(FairseqDataclass):
    dcl_video_alpha: float = 1.0
    dcl_vl_text_alpha: float = 0.5
    dcl_vl_video_alpha: float = 0.5
    dcl_logit_scale: float = 2.5

    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("video_text_pretrain_loss", dataclass=VideoTextPretrainLossConfig)
class VideoTextPretrainLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        dcl_video_alpha,
        dcl_vl_text_alpha,
        dcl_vl_video_alpha,
        dcl_logit_scale,
        label_smoothing=0.0
    ):
        super().__init__(task)
        self.dcl_video_alpha = dcl_video_alpha
        self.dcl_vl_text_alpha = dcl_vl_text_alpha
        self.dcl_vl_video_alpha = dcl_vl_video_alpha
        self.dcl_logit_scale = dcl_logit_scale
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
        video_preserve_ids = sample['net_input']['video_preserve_ids']
        video_mask_indices = sample['net_input']['video_mask_indices']

        vl_text_preserve_ids = sample['net_input']['vl_text_preserve_ids']
        vl_text_mask_indices = sample['net_input']['vl_text_mask_indices']
        vl_video_preserve_ids = sample['net_input']['vl_video_preserve_ids']
        vl_video_mask_indices = sample['net_input']['vl_video_mask_indices']

        with torch.no_grad():
            text_logits, teacher_text_features = model(src_tokens=src_tokens, encoder_type='text')
        video_logits, _ = model(
            src_videos=src_videos, video_padding_masks=video_padding_masks, encoder_type='video',
        )
        text_logits_all = gather_without_grad(text_logits) if dist.is_initialized() else text_logits.data
        video_logits_all = gather_without_grad(video_logits) if dist.is_initialized() else video_logits.data
        with torch.no_grad():
            teacher_vl_text_features, teacher_vl_video_features = model(
                src_tokens=src_tokens, src_videos=src_videos, video_padding_masks=video_padding_masks,
                encoder_type='vl'
            )

        _, _, student_video_features = model(
            src_videos=src_videos,
            video_preserve_ids=video_preserve_ids,
            video_padding_masks=video_padding_masks,
            encoder_type='video'
        )
        student_vl_text_features, _, student_vl_video_features = model(
            src_tokens=src_tokens, text_preserve_ids=vl_text_preserve_ids,
            src_videos=src_videos, video_padding_masks=video_padding_masks, video_preserve_ids=vl_video_preserve_ids,
            encoder_type='vl'
        )

        logit_scale_exp = model(return_logit_scale=True)

        text_padding_masks = src_tokens.eq(1)
        video_padding_masks = video_padding_masks[:, 1:]
        dcl_video_loss = self.compute_dcl_loss(
            student_video_features, teacher_vl_video_features, video_mask_indices,
            padding_masks=video_padding_masks
        )
        dcl_vl_text_loss = self.compute_dcl_loss(
            student_vl_text_features, teacher_vl_text_features, vl_text_mask_indices,
            padding_masks=text_padding_masks
        )
        dcl_vl_video_loss = self.compute_dcl_loss(
            student_vl_video_features, teacher_vl_video_features, vl_video_mask_indices,
            padding_masks=video_padding_masks
        )

        vtc_loss, v2t_ncorrect, t2v_ncorrect = self.compute_vtc_loss(
            video_logits, text_logits,
            video_logits_all, text_logits_all,
            logit_scale_exp
        )

        loss = vtc_loss + \
               self.dcl_video_alpha * dcl_video_loss + \
               self.dcl_vl_text_alpha * dcl_vl_text_loss + self.dcl_vl_video_alpha * dcl_vl_video_loss
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "vtc_loss": vtc_loss.data,
            "dcl_video_loss": dcl_video_loss.data,
            "dcl_vl_text_loss": dcl_vl_text_loss.data,
            "dcl_vl_video_loss": dcl_vl_video_loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
            "v2t_ncorrect": v2t_ncorrect,
            "t2v_ncorrect": t2v_ncorrect,
            "logit_scale_exp": logit_scale_exp
        }
        return loss, sample_size, logging_output

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
        v2t_loss = adjust_label_smoothed_nll_loss(log_sim_v2t, targets)
        t2v_loss = adjust_label_smoothed_nll_loss(log_sim_t2v, targets)
        vtc_loss = (v2t_loss + t2v_loss) / 2

        with torch.no_grad():
            v2t_preds = sim_v2t.argmax(dim=1)
            t2v_preds = sim_t2v.argmax(dim=1)
            v2t_ncorrect = (v2t_preds == targets).float().sum()
            t2v_ncorrect = (t2v_preds == targets).float().sum()

        return vtc_loss, v2t_ncorrect, t2v_ncorrect

    def compute_dcl_loss(self, student_features, teacher_features, mask_indices, padding_masks=None):
        embed_dim = student_features.size(-1)
        teacher_features = teacher_features.detach()
        student_out = student_features[:, 1:, :].reshape(-1, embed_dim)
        teacher_out = teacher_features[:, 1:, :].reshape(-1, embed_dim)
        mask_indices = mask_indices[:, 1:].flatten()
        if padding_masks is not None:
            non_padding_mask_indices = torch.nonzero((~padding_masks).flatten(), as_tuple=False).flatten()
            student_out = student_out[non_padding_mask_indices]
            teacher_out = teacher_out[non_padding_mask_indices]
            mask_indices = mask_indices[non_padding_mask_indices]

        indices = torch.nonzero(mask_indices, as_tuple=False).flatten()
        targets = torch.arange(student_out.size(0)).to(student_out.device)[indices]
        orig_type = student_out.dtype
        mask_student_out = F.normalize(student_out[indices].float(), dim=1).to(orig_type)
        teacher_out = F.normalize(teacher_out.float(), dim=1).to(orig_type)

        sim_stu2tea = self.dcl_logit_scale * mask_student_out @ teacher_out.t()
        log_sim_stu2tea = utils.log_softmax(sim_stu2tea, dim=-1).type_as(sim_stu2tea)
        loss = adjust_label_smoothed_nll_loss(log_sim_stu2tea, targets, self.label_smoothing)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        vtc_loss_sum = sum(log.get("vtc_loss", 0) for log in logging_outputs)
        dcl_video_loss_sum = sum(log.get("dcl_video_loss", 0) for log in logging_outputs)
        dcl_vl_text_loss_sum = sum(log.get("dcl_vl_text_loss", 0) for log in logging_outputs)
        dcl_vl_video_loss_sum = sum(log.get("dcl_vl_video_loss", 0) for log in logging_outputs)
        logit_scale_exp_sum = sum(log.get("logit_scale_exp", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "vtc_loss", vtc_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_video_loss", dcl_video_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_vl_text_loss", dcl_vl_text_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dcl_vl_video_loss", dcl_vl_video_loss_sum / sample_size, sample_size, round=3
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
        to True will improve distributed training speed.
        """
        return True
