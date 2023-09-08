# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
from transformers import AutoImageProcessor, EfficientNetModel
import torch
import os
from PIL import Image
import timm
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
#from signjoey.encoders import SwinTransformerEncoder
import torch
import numpy as np
from torchvision.io import read_video
from pathlib import Path
import torchvision

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        #if not isinstance(path, list):
        #    path = [path]

        annotation_file = os.path.join(path, "annotations.gzip")
        samples = {}
        with gzip.open(annotation_file, 'rb') as f:
            annotation = pickle.load(f)
        print("Annotation file loaded successfully from:", annotation_file)
        print("There annotation file has this number of entries: ", len(annotation))

        counter = 0
        for s in annotation:
            seq_id = s["name"]
            seq_id = seq_id.replace("train/", "")
            seq_id = seq_id.replace("dev/", "")
            seq_id = seq_id.replace("test/", "")

            video_path = os.path.join(path, seq_id + ".mp4")
            if Path(video_path).exists() and counter < 1000:
                sign_video, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
                sign_video = sign_video/255
                no_frames = sign_video.shape[0]
                if no_frames >= 100 and no_frames <= 200:  # dont use shorter videos
                    factor = no_frames / 100
                    lower_step = int(np.floor(factor))
                    upper_step = int(np.ceil(factor))
                    decimal = int((factor % 1) * 100)
                    step_1 = lower_step
                    start_1 = lower_step - 1
                    end_1 = (100 - decimal) * step_1
                    step_2 = upper_step
                    start_2 = end_1 - 1 + step_2
                    sign_video_1 = sign_video[start_1:end_1:step_1]  # take every factor-th frame
                    sign_video_2 = sign_video[start_2::step_2]  # take every factor-th frame
                    sign_video = torch.cat((sign_video_1, sign_video_2), 0)
                    counter += 1

                    if seq_id in samples:
                        assert samples[seq_id]["name"] == s["name"]
                        assert samples[seq_id]["signer"] == s["signer"]
                        assert samples[seq_id]["gloss"] == s["gloss"]
                        assert samples[seq_id]["text"] == s["text"]
                        samples[seq_id]["sign"] = torch.cat(
                            [samples[seq_id]["sign"], s["sign"]], axis=1
                        )
                    else:
                        samples[seq_id] = {
                            "name": s["name"],
                            "signer": s["signer"],
                            "gloss": s["gloss"],
                            "text": s["text"],
                            "sign": self.get_embeddings(sign_video),
                        }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)

    def get_embeddings(self, sign_video):
        sign_video = sign_video.to(torch.float32)
        sign_video_resized = torchvision.transforms.functional.resize(sign_video, (224, 224))

        image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
        #image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")

        model = EfficientNetModel.from_pretrained("google/efficientnet-b0")
        #model = EfficientNetModel.from_pretrained("google/efficientnet-b7")

        inputs = image_processor(sign_video_resized, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        outputs = [frame.flatten() for frame in last_hidden_states]
        outputs = torch.from_numpy(np.asarray(outputs))
        
        return outputs
