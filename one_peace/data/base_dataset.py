import os
import math
import logging
import warnings
import re
import torch
import torch.nn.functional as F
import soundfile as sf
from PIL import Image
import av  # PyAV for video processing
import cv2
from fairseq.data import FairseqDataset
import numpy as np

from . import collate_fn

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

class BaseDataset(FairseqDataset):
    def __init__(self, split, dataset, bpe, dictionary):
        self.split = split
        self.dataset = dataset
        self.bpe = bpe
        self.dict = dictionary

        self.bos = dictionary.bos()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        self.dataset_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.realpath(__file__), "../..")),
            "dataset"
        )

        # for audio
        self._features_size_map = {}

    def __len__(self):
        return len(self.dataset)

    def read_image(self, image_path):
        path = os.path.join(self.dataset_dir, image_path)
        return Image.open(path).convert("RGB")

    def read_audio(self, audio_path):
        path = os.path.join(self.dataset_dir, audio_path)
        return sf.read(path, dtype="float32")

    def read_video(self, video_path, num_frames):
        """
        Processes a video file using OpenCV and returns a list of PIL Images.
        If the video cannot be read, returns num_frames blank images.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #print("FRAMES/NUM_FRAMES: ", frame_count, num_frames)

        if frame_count <= 0:
            print(f"Warning: Failed to read video '{video_path}'. Using blank frames instead.")
            frames = [Image.new('RGB', (224, 224)) for _ in range(num_frames)]
            cap.release()
            return frames

        # Evenly sample indices over the video's frames.
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=np.int32)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                # If reading fails, use a blank image.
                frames.append(Image.new('RGB', (224, 224)))
        cap.release()
        
        return frames

    def encode_text(self, text, length=None, use_bpe=True, append_eos=True):
        s = self.dict.encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_eos:
            s = torch.cat([s, torch.LongTensor([self.eos])])
        return s

    def process_text(self, text, max_length=None):
        text = text.lower().lstrip(",.!?*#:;~")
        text = re.sub(
            r"\s{2,}|\t",
            ' ',
            text,
        )
        text = text.rstrip('\n')
        text = text.strip(' ')

        if max_length is not None:
            text = ' '.join(text.split(' ')[:max_length])

        return text

    def audio_postprocess(self, feats, curr_sample_rate, max_duration=15):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != 16000:
            raise Exception(f"sample rate: {curr_sample_rate}, need 16000")

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)

        if feats.size(-1) > curr_sample_rate * max_duration:
            start_idx = 0
            end_idx = start_idx + curr_sample_rate * max_duration
            feats = feats[start_idx:end_idx]
        if feats.size(-1) < curr_sample_rate * 1:
            feats = feats.repeat(math.ceil(curr_sample_rate * 1 / feats.size(-1)))
            feats = feats[:curr_sample_rate * 1]

        return feats

    def _get_mask_indices_dims(self, size, feature_encoder_spec, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data required for the task
        """
        return collate_fn(samples, pad_idx=self.pad)
