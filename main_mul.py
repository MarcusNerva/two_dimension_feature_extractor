#!/usr/bin/python3
# @Author  : MarcusNerva
# @Email   : yehanhua20@mails.ucas.ac.cn
from models import models_factory_mullevel, trans
import cv2
import h5py
import os
import glob
import argparse
import torch
import PIL.Image as Image
import numpy as np
import tqdm

def extract_frames(video_path, stride):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        raise Exception('Can not open {}'.format(video_path))

    frames = []
    while True:
        ret, item = cap.read()
        if ret is False:
            break
        img = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
        frames.append(img)
    cap.release()
    frame_count = len(frames)
    indices = list(range(8, frame_count - 7, stride))
    frame_list = [frames[i] for i in indices]
    return frame_list

def extract_features(model, images):
    with torch.no_grad():
        level_0s, level_1s, level_2s, level_3s = [], [], [], []
        for img in images:
            img = trans(img).unsqueeze(0).to(device)
            feats = model(img)
            feats = [item.squeeze(0).cpu().numpy() for item in feats]  # List[(C, H, W), ...]
            assert len(feats) == 4
            level_0s.append(feats[0])
            level_1s.append(feats[1])
            level_2s.append(feats[2])
            level_3s.append(feats[3])
        level_0s = np.array(level_0s).astype(float)
        level_1s = np.array(level_1s).astype(float)
        level_2s = np.array(level_2s).astype(float)
        level_3s = np.array(level_3s).astype(float)
    return level_0s, level_1s, level_2s, level_3s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnext101_32x8d', help='which kind of backbone would you like to choose')
    parser.add_argument('--video_dir', type=str, default='-1')
    parser.add_argument('--extension', type=str, default='-1')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='-1')
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    video_dir = args.video_dir
    extension = args.extension
    stride = args.stride
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    visual_feats_save_path = './results/{}.hdf5'.format(dataset_name)
    assert video_dir != '-1', 'Please set video dir!'
    assert extension != '-1', 'Please set video extension!'
    assert dataset_name != '-1', 'Please set dataset name!'

    model = models_factory_mullevel(model_name)
    model = model.to(device)
    model.eval()
    video_path_list = glob.glob(os.path.join(video_dir, '*.{}'.format(extension)))

    with h5py.File(visual_feats_save_path, 'w') as f:
        for video_path in tqdm.tqdm(video_path_list):
            video_base_path = os.path.basename(video_path)
            video_id = video_base_path.split('.')[0]
            frames = extract_frames(video_path, stride)
            levels_feats = extract_features(model, frames)
            f[video_id] = levels_feats

    print('=' * 25, 'finished! results are save in ./results/', '=' * 25)

