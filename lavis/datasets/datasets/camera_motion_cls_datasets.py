"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import json
import re
from PIL import Image
from lavis.datasets.datasets.read_video import read_video
import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.datasets.dataloader_utils import get_frame_indices
class CameraMotionCLSDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, csv_path, num_frames, prompt='', split='train'):
        self.csv_path = csv_path
        self.fps = 10
        self.data = {}
        df = pd.read_csv(self.csv_path)
        for video_id in df:
            video_path = df[video_id]['video_path']
            label = df[video_id]['pose_traj_class']
            label_after_process = text_processor(label)
            assert label == label_after_process, "{} not equal to {}".format(label, label_after_process)
            self.data[video_id] = {
                'video_id': video_id, 
                'label': label_after_process,
                'video_path': video_path
            }
        
        self.video_id_list = list(self.data.keys())
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt


    def __getitem__using_images(self, index):
        video_id = self.video_id_list[index]
        ann = self.data[video_id]
        # Divide the range into num_frames segments and select a random index from each segment
        selected_frame_indices = self.get_frame_indices(ann['frame_length'], self.num_frames)

        frame_list = []
        # logger = logging.getLogger()
        # print(f"video_id: {video_id}, selected_frame_index: {selected_frame_index}")
        for frame_index in selected_frame_indices:
            frame = Image.open(os.path.join(self.vis_root, video_id, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(self.prompt)
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_id,
        }

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        video_info = self.data[video_id]
        video_path = video_info['video_path']
        video, vinfo = read_video(video_path)
        # torch.Size([159, 3, 720, 1280])
        
        
        video = self.vis_processor(video)
        text_input = self.text_processor(self.prompt)
        caption = self.text_processor(video_info['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_id,
        }
        
    def __len__(self):
        return len(self.video_id_list)

class CameraMotionCLSEvalDataset(CameraMotionCLSDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, 
                 num_frames, prompt, split='val'):
        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val')

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.data[video_id]

        # Divide the range into num_frames segments and select a random index from each segment
        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, video_id, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(self.prompt)
        
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_id,
        }
