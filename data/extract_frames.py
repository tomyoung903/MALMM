import os
import threading
import shutil
from tqdm import tqdm

FFMPEG_PATH = '/home/tom/.conda/envs/opensora/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2'

import pdb
# /home/tom/Open-Sora-dev/tools/scoring/MA-LMM/data/breakfast/frames/P39-cam02-P39_scrambledegg/frame000006.jpg
FPS = 10

def split_list(l, n):
    """Yield successive n-sized chunks from a list l."""
    length = len(l)
    chunk_size = length // n + 1
    for i in range(0, length, chunk_size):
        yield l[i:i + chunk_size]

def extract_frames(video_name):
    video_id = video_name.split('.')[0]
    os.makedirs("{}/{}".format(dst_base_dir, video_id), exist_ok=True)
    cmd = '{} -i \"{}/{}\" -vf scale=-1:256 -pix_fmt yuvj422p -q:v 1 -r {} -y \"{}/{}/frame%06d.jpg\"'.format(
        FFMPEG_PATH, src_base_dir, video_name, FPS, dst_base_dir, video_id, FPS)
    print(cmd)
    # exit()
    os.system(cmd)

def target(full_id_list):
    for video_id in tqdm(full_id_list):
        extract_frames(video_id)

if __name__ == '__main__':
    dataset = 'breakfast'
    # src_base_dir = f'{dataset}/videos/P03/cam01'
    # dst_base_dir = f'{dataset}/frames/P03/cam01'
    # /home/tom/Open-Sora-dev/tools/scoring/MA-LMM/data/subdirs.txt
    subdirs = open("/home/tom/Open-Sora-dev/tools/scoring/MA-LMM/data/subdirs.txt").read().splitlines()
    for subdir in subdirs:    
        dataset = 'breakfast'
        src_base_dir = subdir
        dst_base_dir = src_base_dir.replace('videos', 'frames')
        
        full_name_list = []
        for video_name in os.listdir(src_base_dir):
            if not os.path.exists("{}/{}".format(src_base_dir, video_name)):
                continue
            full_name_list.append(video_name)

        full_name_list.sort()
        NUM_THREADS = 4
        splits = list(split_list(full_name_list, NUM_THREADS))
        
        threads = []
        for i, split in enumerate(splits):
            thread = threading.Thread(target=target, args=(split,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
            
