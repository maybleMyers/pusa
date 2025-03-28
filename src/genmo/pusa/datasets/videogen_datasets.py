import os
import re
import json
import torch
import decord
import torchvision
import numpy as np
import ipdb

from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple

from genmo.mochi_preview.vae.models import Encoder, add_fourier_features

class_labels_map = None
cls_sample_cnt = None

class_labels_map = None
cls_sample_cnt = None


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

def process_caption(caption):
    # Split the caption into words
    words = caption.split()
    
    if len(words) >= 3:
        # Join the remaining words after removing first three
        new_caption = ' '.join(words[3:])
        
        # Remove comma if it's the first character
        if new_caption and new_caption[0] == ',':
            new_caption = new_caption[1:].strip()
        
        # Capitalize the first letter
        if new_caption:
            new_caption = new_caption[0].upper() + new_caption[1:]
            
        return new_caption
    return caption
    
class VIDGEN(torch.utils.data.Dataset):
    """Load the VIDGEN video files and their corresponding captions
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs['data_path'] if isinstance(configs, dict) else configs.data_path
        
        # Load video files
        video_dir = os.path.join(self.data_path, 'videos')
        # self.video_lists = get_filelist(video_dir)[:100]
        self.video_lists = get_filelist(video_dir)
        
        # Load caption annotations
        caption_file = os.path.join(self.data_path, 'VidGen_1M_video_caption.json')
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
        
        # Create a mapping from video ID to caption
        self.vid_to_caption = {}
        for item in caption_data:
            self.vid_to_caption[item['vid']] = item['caption']
            
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs['num_frames'] if isinstance(self.configs, dict) else self.configs.num_frames
        self.v_decoder = DecordInit()

    def __getitem__(self, index):
        # Get video path
        video_path = self.video_lists[index]
        
        # Extract video ID from filename (remove .mp4 extension)
        video_id = os.path.basename(video_path)[:-4]
        
        # Get corresponding caption
        caption = self.vid_to_caption.get(video_id, "")
        if not caption:
            print(f"Warning: No caption found for video {video_id}")
        else:
            caption = process_caption(caption)

        # Load and process video frames using decord instead of torchvision
        vr = self.v_decoder(video_path)
        total_frames = len(vr)
        
        if total_frames >= self.target_video_len:
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
        else:
            # Temporal Resampling to handle fewer frames
            frame_indices = self.resample_frames(total_frames, self.target_video_len)
        
        # Read frames using decord
        video = vr.get_batch(frame_indices).asnumpy()  # Returns (T,H,W,C)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # Convert to (T,C,H,W)
        
        video = self.transform(video)  # T C H W
        video = rearrange(video, 't c h w -> c t h w')  # Convert to C T H W

        video = video.float() / 127.5 - 1.0
        
        
        # Convert to float in [-1, 1] range if not already done in transform
        # if video.dtype == torch.uint8:
        #     video = video.float() / 127.5 - 1.0

        return {'video': video, 'caption': caption, 'video_path': video_path}

    def resample_frames(self, total_frames, target_frames):
        """
        Resample frame indices to match the target frame count.
        This function proportionally duplicates frames when
        the total_frames is less than target_frames.
        """
        if total_frames == 0:
            raise ValueError("Video contains no frames.")
        # Generate evenly spaced indices
        ratio = target_frames / total_frames
        # Create a list of indices with proportional duplication
        resampled_indices = []
        for i in range(total_frames):
            # Determine how many times to duplicate each frame
            duplicates = int(np.floor(ratio))
            resampled_indices.extend([i] * duplicates)
            # Handle remaining frames based on the fractional part
            if np.random.rand() < (ratio - duplicates):
                resampled_indices.append(i)
        # If we have fewer frames due to flooring, pad with the last frame
        while len(resampled_indices) < target_frames:
            resampled_indices.append(total_frames - 1)
        # Trim to the exact target_frames
        resampled_indices = resampled_indices[:target_frames]
        return resampled_indices
        
            
    def __len__(self):
        return len(self.video_lists)


if __name__ == '__main__':

    import argparse
    import video_transforms
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=163, help="Number of frames.")
    parser.add_argument("--width", type=int, default=848, help="Width of the video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the video.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval.")
    parser.add_argument("--data-path", type=str, default="/home/dyvm6xra/dyvm6xrauser02/data/vidgen1m")
    config = parser.parse_args()

    temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_VIDGEN = transforms.Compose([
        # video_transforms.ToTensorVideo(),  # converting from numpy arrays to PyTorch tensors, the pixel values are normalized to the range [0, 1] by dividing by 255
        video_transforms.RandomHorizontalFlipVideo(),
        video_transforms.ResizeVideo((config.height, config.width)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    ffs_dataset = VIDGEN(config, transform=transform_VIDGEN, temporal_sample=temporal_sample)
    ffs_dataloader = Data.DataLoader(
        dataset=ffs_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1
    )

    
    # for i, video_data in enumerate(ffs_dataloader):
    for video_data in ffs_dataloader:
        print(type(video_data))
        video = video_data['video']
        caption = video_data['caption']
        # ipdb.set_trace()
        print(video.shape)
        print(caption)

        print(f"Mean Intensity = {video.mean().item():.4f}, Standard Deviation = {video.std().item():.4f}, max ={video.max().item():.4f}, min ={video.min().item():.4f}")
        # print(video_data[2])

        # for i in range(16):
        #     img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
        #     print('Label: {}'.format(video_data[1]))
        #     print(img0.shape)
        #     img0 = Image.fromarray(np.uint8(img0 * 255))
        #     img0.save('./img{}.jpg'.format(i))
        # exit()