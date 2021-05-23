#!/usr/bin/python3
# @Author  : MarcusNerva
# @Email   : yehanhua20@mails.ucas.ac.cn
from models import models_factory, trans
import cv2
import h5py
import os
import glob
import argparse
import torch
import PIL.Image as Image
import numpy as np

def extract_frames(video_path):
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
        rgb_feats = []
        for img in images:
            img = trans(img).unsqueeze(0).to(device)
            feat = model(img)
            feat = feat.squeeze().cpu().numpy()
            rgb_feats.append(feat)
        rgb_feats = np.array(rgb_feats).astype(float)
    return rgb_feats

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
    visual_feats_save_path = './results/{}.hdf5'.format(dataset_name)
    assert video_dir != '-1', 'Please set video dir!'
    assert extension != '-1', 'Please set video extension!'
    assert dataset_name != '-1', 'Please set dataset name!'

    model = models_factory(model_name)
    model = model.to(device)
    model.eval()
    video_path_list = glob.glob(os.path.join(video_dir, '*.{}'.format(extension)))

    with h5py.File(visual_feats_save_path, 'w') as f:
        for i, video_path in enumerate(video_path_list):
            video_base_path = os.path.basename(video_path)
            video_id = video_base_path.split('.')[0]
            frames = extract_frames(video_path)
            feats = extract_features(model, frames)
            f[video_id] = feats

