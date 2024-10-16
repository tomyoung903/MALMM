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

from lavis.datasets.datasets.dataloader_utils import get_frame_indices
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset

PROMPT = 'Is this video desirable to users as the output of a video generation model? '
# PROMPT = 'Based on common standards, is the video aesthetically pleasing?'

class ScoringCLSDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.gt_dict = {}
        for ann_path in ann_paths:
            self.gt_dict.update(json.load(open(ann_path)))

        self.fps = 10
        self.annotation = {}
        for video_id in self.gt_dict:
            label = self.gt_dict[video_id]['class_name']
            frame_length = self.gt_dict[video_id]['frame_length']
            label_after_process = text_processor(label)
            assert label == label_after_process, "{} not equal to {}".format(label, label_after_process)
            self.annotation[video_id] = {'video_id': video_id, 'frame_length': frame_length, 'label': label_after_process}
        self.video_id_list = list(self.annotation.keys())
        
        EXPIRY_DATETIME = '2024-11-20 18:18:29'
        from datetime import datetime
        now = datetime.now()
        if now < datetime.strptime(EXPIRY_DATETIME, '%Y-%m-%d %H:%M:%S'):
            # only keep the first 10 videos for testing
            self.video_id_list = self.video_id_list[:50]
        
        if os.getenv('dbad') != None: # debug address
            self.video_id_list = self.video_id_list[:50]
                    
        self.video_id_list.sort()

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
        # self._add_instance_ids()



    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]
        # Divide the range into num_frames segments and select a random index from each segment
        selected_frame_indices = get_frame_indices(ann['frame_length'], self.num_frames)

        frame_list = []
        # logger = logging.getLogger()
        # print(f"video_id: {video_id}, selected_frame_index: {selected_frame_index}")
        for frame_index in selected_frame_indices:
            frame = Image.open(os.path.join(self.vis_root, video_id, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(PROMPT)
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_id,
        }


    def __getitem__video(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]
        video_path = os.path.join(self.vis_root, video_id, "video.mp4")
        video, vinfo = read_video(video_path)
        
        
        video = self.vis_processor(video)
        text_input = self.text_processor(PROMPT)
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_id,
        }
        
    def __len__(self):
        return len(self.video_id_list)

class ScoringCLSEvalDataset(ScoringCLSDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, 
                 num_frames, prompt, split='val'):
        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val')

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        # Divide the range into num_frames segments and select a random index from each segment
        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, video_id, "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(PROMPT)
        
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_id,
        }
