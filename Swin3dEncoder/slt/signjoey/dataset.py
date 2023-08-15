# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
from torchvision.io import read_video
import gzip
import torch
import os
import numpy as np
from pathlib import Path


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

        print("path for data", path)
        annotation_file = os.path.join(path, "annotations.gzip")
        print("path for annotation file", annotation_file)
        samples = {}
        with gzip.open(annotation_file, 'rb') as f:
            annotations = pickle.load(f)
        print("Annotation file loaded successfully from:", annotation_file)
        print("There annotation file has this number of entries: ", len(annotations))

        print_one_sample = True
        counter = 0
        for s in annotations:
            seq_id = s["name"]
            seq_id = seq_id.replace("train/", "")
            seq_id = seq_id.replace("dev/", "")
            seq_id = seq_id.replace("test/", "")

            video_path = os.path.join(path, seq_id + ".mp4")

            if Path(video_path).exists() and counter < 100:
                sign_video, _, _ = read_video(video_path, output_format="THWC", pts_unit="sec")
                print(sign_video.shape)
                no_frames = sign_video.shape[0]
                if no_frames >= 100: # dont use shorter videos
                    factor = no_frames/100
                    print("downsampling factor is ", factor)
                    #sign_video_np = np.zeros((no_frames//factor, sign_video.shape[1], sign_video.shape[2], 3))
                    lower_step = int(np.floor(factor))
                    print(lower_step)
                    upper_step = int(np.ceil(factor))
                    print(upper_step)
                    decimal = int((factor % 1)*100)
                    print(decimal)
                    step_1 = lower_step
                    print(step_1)
                    start_1 = lower_step - 1
                    print(start_1)
                    end_1 = (100-decimal) * step_1
                    print(end_1)
                    step_2 = upper_step
                    print(step_2)
                    start_2 = end_1 - 1 + step_2
                    print(start_2)
                    #print("downsampled video frames", no_frames//factor)
                    sign_video_1 = sign_video[start_1:end_1:step_1]  # take every factor-th frame
                    sign_video_2 = sign_video[start_2::step_2]   # take every factor-th frame
                    sign_video = torch.cat((sign_video_1, sign_video_2), 0)
                    #sign_video = torch.from_numpy(sign_video_np)
                    print("downsampled video frames", sign_video.shape[0])
                    print(path)
                    print(counter)
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
                            "sign": sign_video,
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
