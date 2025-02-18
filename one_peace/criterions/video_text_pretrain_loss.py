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
    tensors_gather = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

@dataclass
class VideoTextPretrainLossConfig(FairseqDataclass):
    dcl_video_alpha: float = 1.0
    dcl_vid_text_alpha: float = 0.5
    dcl_vid_video_alpha: float = 0.5
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
        dcl_vid_text_alpha,
        dcl_vid_video_alpha,
        dcl_logit_scale,
        label_smoothing=0.0
    ):
        super().__init__(task)
        self.dcl_video_alpha = dcl_video_alpha
        self.dcl_vid_text_alpha = dcl_vid_text_alpha
        self.dcl_vid_video_alpha = dcl_vid_video_alpha
        self.dcl_logit_scale = dcl_logit_scale
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        """
        Computes the loss for video–text pretraining.
        Returns:
            loss, sample_size, logging_output
        """
        src_tokens = sample['net_input']['src_tokens']
        src_videos = sample['net_input']['src_videos']

        video_preserve_ids = sample['net_input']['video_preserve_ids']
        video_mask_indices = sample['net_input']['video_mask_indices']

        vid_text_preserve_ids = sample['net_input']['vid_text_preserve_ids']
        vid_text_mask_indices = sample['net_input']['vid_text_mask_indices']
        vid_video_preserve_ids = sample['net_input']['vid_video_preserve_ids']
        vid_video_mask_indices = sample['net_input']['vid_video_mask_indices']

        # Get text logits and teacher text features (using encoder_type 'text').
        with torch.no_grad():
            text_logits, teacher_text_features = model(src_tokens=src_tokens, encoder_type='text')
        # Get video logits from the video adapter (encoder_type 'video').
        video_logits, _ = model(src_videos=src_videos, encoder_type='video')
        
        # Gather logits from all GPUs if using distributed training.
        text_logits_all = (gather_without_grad(text_logits)
                           if dist.is_initialized() else text_logits.data)
        video_logits_all = (gather_without_grad(video_logits)
                            if dist.is_initialized() else video_logits.data)
        
        # Get teacher features for alignment (using encoder_type 'al').
        with torch.no_grad():
            teacher_vid_text_features, teacher_vid_video_features = model(
                src_tokens=src_tokens, src_videos=src_videos, encoder_type='vid'
            )
        
        # Get student video features.
        _, _, _, student_video_features = model(
            src_videos=src_videos,
            video_preserve_ids=video_preserve_ids,
            encoder_type='video'
        )
        # Get student features from the alignment branch.
        student_vid_text_features, _, _, student_vid_video_features = model(
            src_tokens=src_tokens, text_preserve_ids=vid_text_preserve_ids,
            src_videos=src_videos, video_preserve_ids=vid_video_preserve_ids,
            encoder_type='vid'
        )
        
        # Get logit scale.
        logit_scale_exp = model(return_logit_scale=True)
        
        # Contrastive loss between text and video logits.
        # Assume each GPU holds a shard; we use rank-based indexing.
        slice_id = dist.get_rank() if dist.is_initialized() else 0
        bsz = video_logits.size(0)
        start_idx = bsz * slice_id
        end_idx = start_idx + bsz
        targets = torch.arange(start_idx, end_idx).to(video_logits.device)
        
        sim_v2t = logit_scale_exp * video_logits @ text_logits_all.t()
        sim_t2v = logit_scale_exp * text_logits @ video_logits_all.t()
        log_sim_v2t = F.log_softmax(sim_v2t, dim=-1)
        log_sim_t2v = F.log_softmax(sim_t2v, dim=-1)
        v2t_loss = adjust_label_smoothed_nll_loss(log_sim_v2t, targets, self.label_smoothing)
        t2v_loss = adjust_label_smoothed_nll_loss(log_sim_t2v, targets, self.label_smoothing)
        atc_loss = (v2t_loss + t2v_loss) / 2
        
        # Compute DCL (feature-level) losses.
        text_padding_masks = src_tokens.eq(1)
        dcl_video_loss = self.compute_dcl_loss(
            student_video_features, teacher_vid_video_features, video_mask_indices,
            padding_masks=None  # Assuming video inputs are uniformly processed.
        )
        dcl_vid_text_loss = self.compute_dcl_loss(
            student_vid_text_features, teacher_vid_text_features, vid_text_mask_indices,
            padding_masks=text_padding_masks
        )
        dcl_vid_video_loss = self.compute_dcl_loss(
            student_vid_video_features, teacher_vid_video_features, vid_video_mask_indices,
            padding_masks=None
        )

        atc_loss, a2t_ncorrect, t2a_ncorrect = self.compute_atc_loss(
            video_logits, text_logits,
            video_logits_all, text_logits_all,
            logit_scale_exp
        )
        
        loss = atc_loss + \
               self.dcl_video_alpha * dcl_video_loss + \
               self.dcl_vid_text_alpha * dcl_vid_text_loss
        sample_size = 1 
        logging_output = {
            "loss": loss.data,
            "atc_loss": atc_loss.data,
            "dcl_video_loss": dcl_video_loss.data,
            "dcl_vid_text_loss": dcl_vid_text_loss.data,
            "dcl_vid_video_loss": dcl_vid_video_loss.data,
            "nsentences": sample['nsentences'],
            "sample_size": sample_size,
            "a2t_ncorrect": a2t_ncorrect,
            "t2a_ncorrect": t2a_ncorrect,
            "logit_scale_exp": logit_scale_exp
        }
        return loss, sample_size, logging_output

    def compute_dcl_loss(self, student_features, teacher_features, mask_indices, padding_masks=None):
        embed_dim = student_features.size(-1)
        teacher_features = teacher_features.detach()
        # Remove the CLS token. Assume student_features shape: [B, seq_len, D]
        # After removal, we expect a length of (seq_len - 1)
        desired_length = student_features.size(1)
        student_out = student_features[:, 1:, :].reshape(-1, embed_dim)
        teacher_out = teacher_features[:, 1:, :].reshape(-1, embed_dim)
        
        if mask_indices is not None:
            # Adjust mask_indices to have shape [B, desired_length]
            if mask_indices.size(1) > desired_length:
                mask_indices = mask_indices[:, :desired_length]
            elif mask_indices.size(1) < desired_length:
                pad_length = desired_length - mask_indices.size(1)
                pad = torch.zeros(mask_indices.size(0), pad_length, dtype=mask_indices.dtype, device=mask_indices.device)
                mask_indices = torch.cat([mask_indices, pad], dim=1)
            # Remove the CLS token's mask (assumed to be at index 0)
            mask_indices = mask_indices[:, 1:desired_length]
            mask_indices = mask_indices.flatten()
            
            if padding_masks is not None:
                if padding_masks.size(1) > desired_length:
                    padding_masks = padding_masks[:, :desired_length]
                elif padding_masks.size(1) < desired_length:
                    pad_length = desired_length - padding_masks.size(1)
                    pad = torch.zeros(padding_masks.size(0), pad_length, dtype=padding_masks.dtype, device=padding_masks.device)
                    padding_masks = torch.cat([padding_masks, pad], dim=1)
                # Remove the CLS token's entry from padding_masks as well.
                padding_masks = padding_masks[:, 1:desired_length]
                non_padding = torch.nonzero((~padding_masks).flatten(), as_tuple=False).flatten()
                student_out = student_out[non_padding]
                teacher_out = teacher_out[non_padding]
                mask_indices = mask_indices[non_padding]
            
            selected = torch.nonzero(mask_indices, as_tuple=False).flatten()
            targets = torch.arange(student_out.size(0)).to(student_out.device)[selected]
            student_norm = F.normalize(student_out[selected].float(), dim=1)
            teacher_norm = F.normalize(teacher_out.float(), dim=1)
            sim = self.dcl_logit_scale * student_norm @ teacher_norm.t()
            log_sim = F.log_softmax(sim, dim=-1)
            loss = adjust_label_smoothed_nll_loss(log_sim, targets, self.label_smoothing)
            return loss
        else:
            return torch.tensor(0.0, device=student_features.device)



    def compute_atc_loss(self, audio_logits, text_logits, audio_logits_all, text_logits_all, logit_scale_exp):
        slice_id = dist.get_rank() if dist.is_initialized() else 0
        bsz = audio_logits.size(0)
        start_idx = bsz * slice_id
        end_idx = start_idx + bsz
        targets = torch.arange(start_idx, end_idx).to(audio_logits.device)

        sim_a2t = logit_scale_exp * audio_logits @ text_logits_all.t()
        sim_t2a = logit_scale_exp * text_logits @ audio_logits_all.t()
        log_sim_a2t = utils.log_softmax(sim_a2t, dim=-1).type_as(sim_a2t)
        log_sim_t2a = utils.log_softmax(sim_t2a, dim=-1).type_as(sim_t2a)
        a2t_loss = adjust_label_smoothed_nll_loss(log_sim_a2t, targets)
        t2a_loss = adjust_label_smoothed_nll_loss(log_sim_t2a, targets)
        atc_loss = (a2t_loss + t2a_loss) / 2

        with torch.no_grad():
            a2t_preds = sim_a2t.argmax(dim=1)
            t2a_preds = sim_t2a.argmax(dim=1)
            a2t_ncorrect = (a2t_preds == targets).float().sum()
            t2a_ncorrect = (t2a_preds == targets).float().sum()

        return atc_loss, a2t_ncorrect, t2a_ncorrect

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        atc_loss_sum = sum(log.get("atc_loss", 0) for log in logging_outputs)
        dcl_video_loss_sum = sum(log.get("dcl_video_loss", 0) for log in logging_outputs)
        dcl_vid_text_loss_sum = sum(log.get("dcl_vid_text_loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("atc_loss", atc_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("dcl_video_loss", dcl_video_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("dcl_vid_text_loss", dcl_vid_text_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("nsentences", nsentences, 1, round=3)
        metrics.log_scalar("sample_size", sample_size, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
