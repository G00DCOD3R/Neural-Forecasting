# Writes video_frames.npy, uint8 [T, H, W]: the stimulus frame at every row of X_features.npy.
# Clips come in different resolutions (144x256, 126x216, 90x160), so all are resampled to H x W.
# PCA is fitted later on training rows (exogenous.py).
#   python build_video_features.py --processed processed/session_4_7_cave_proofread
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from reader import MicronsReader


# [F, H, W] uint8 -> [F, *size] uint8, area averaging
def resample(clip, size):
    x = F.interpolate(torch.from_numpy(clip).float()[:, None], size=size, mode='area')[:, 0]
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


def build_video_features(processed, h5_path='dataset/microns.h5', size=(36, 64)):
    rows = pd.read_csv(os.path.join(processed, 'conditions.csv'))
    trials = rows.groupby('trial_id', sort=True)['condition_hash'].agg(['first', 'size'])
    clips = {}
    with MicronsReader(h5_path) as reader:
        for h in trials['first'].unique():
            clip, _ = reader.get_video_data(h)
            clips[h] = resample(clip, size)

    frames = []
    for trial_id, (h, n) in trials.iterrows():
        assert clips[h].shape[0] == n, f"trial {trial_id}: clip has {clips[h].shape[0]} frames, trial has {n} rows"
        frames.append(clips[h])
    frames = np.concatenate(frames)
    assert len(frames) == len(rows)
    np.save(os.path.join(processed, 'video_frames.npy'), frames)
    print(f"{processed}: video_frames.npy {frames.shape} from {len(clips)} clips")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract stimulus frames for processed sessions')
    parser.add_argument('--processed', nargs='+', default=['processed/session_4_7_cave_proofread',
                                                           'processed/session_4_7_cave_all'])
    parser.add_argument('--h5', default='dataset/microns.h5')
    parser.add_argument('--size', type=int, nargs=2, default=(36, 64), help='output height and width')
    args = parser.parse_args()
    for p in args.processed:
        build_video_features(p, args.h5, tuple(args.size))
