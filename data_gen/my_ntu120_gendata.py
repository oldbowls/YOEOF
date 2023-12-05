import sys

sys.path.extend(['../'])

import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from data_gen.preprocess import pre_normalization
import random

# NTU RGB+D Skeleton 120 Configurations: https://arxiv.org/pdf/1905.04757.pdf
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]

# Even numbered setups (2,4,...,32) used for training
training_setups = set(range(2, 33, 2))

max_body_true = 2
max_body_kinect = 4

num_joint = 25
max_frame = 60


def read_skeleton_filter(path):
    with open(path, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(path, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(path)
    # Create single skeleton tensor: (M, T, V, C)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # To (C,T,V,M)
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, gap, ignored_sample_path=None, benchmark='xview', part='eval', random_flag=False):
    ignored_samples = []
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    sample_name = []
    sample_label = []
    for filename in sorted(os.listdir(data_path)):
        if filename in ignored_samples:
            continue

        setup_loc = filename.find('S')
        subject_loc = filename.find('P')
        action_loc = filename.find('A')
        setup_id = int(filename[(setup_loc + 1):(setup_loc + 4)])
        subject_id = int(filename[(subject_loc + 1):(subject_loc + 4)])
        action_class = int(filename[(action_loc + 1):(action_loc + 4)])

        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xset':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError(f'Unsupported benchmark: {benchmark}')

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError(f'Unsupported dataset part: {part}')

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)  # to 0-indexed

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    if gap == 0:
        fp = np.zeros((1, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    else:
        fp = np.zeros((2, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
        last_frames = np.zeros((1, 3, 1, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect,
                        num_joint=num_joint)

        len = data.shape[1] - max_frame - gap + 1
        if len <= 0:
            print('continue')
            continue
        if random_flag:
            if gap == 0:  # 滑动窗口，逐帧给teacher模型用来生成测试数据集
                select_sample_index = random.randint(0, len - 1)
                fp[0, :, :, :, :] = data[:, select_sample_index:select_sample_index + max_frame, :, :]
            else:
                select_sample_index = random.randint(0, len - 1)
                fp[0, :, 0:60, :, :] = data[:, select_sample_index:select_sample_index + max_frame, :, :]
                fp[1, :, 0:60, :, :] = data[:, select_sample_index + gap:select_sample_index + max_frame + gap, :, :]
                last_frames[0, :, :, :, :] = data[:,
                                             select_sample_index + max_frame + gap - 1:select_sample_index + max_frame + gap,
                                             :, :]
            fp_temp = pre_normalization(fp)
            last_frames_temp = pre_normalization(last_frames)
            if isinstance(fp_temp, str):
                print("data continue")
                continue
            if isinstance(last_frames_temp, str):
                print("last frame continue")
                continue
            data = {'data': fp_temp, 'last_frame': last_frames_temp}
            np.save(r'{}\{}_data_joint_{}_{}.npy'.format(out_path, part, s, str(select_sample_index)), data)
        else:
            for j in range(0, len):
                if gap == 0:  # 滑动窗口，逐帧给teacher模型用来生成测试数据集
                    fp[0, :, :, :, :] = data[:, j:j + max_frame, :, :]
                else:
                    fp[0, :, :, :, :] = data[:, j:j + max_frame, :, :]
                    fp[1, :, :, :, :] = data[:, j + gap:j + max_frame + gap, :, :]
                    last_frames[0, :, :, :, :] = data[:,
                                                 j + max_frame + gap - 1:j + max_frame + gap, :, :]
                    last_frames = pre_normalization(last_frames)
                    if isinstance(last_frames, str):
                        print("last frame continue")
                        continue
                fp_temp = pre_normalization(fp)
                if isinstance(fp_temp, str):
                    print("data continue")
                    continue
                np.save('{}/{}_data_joint_{}_{}.npy'.format(out_path, part, s, str(j)), fp_temp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    # parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--data_path',
                        default=[r'H:\ntu_paper_data\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons',
                                 r'H:\ntu_paper_data\nturgbd_skeletons_s018_to_s032'])
    # parser.add_argument('--data_path',
    #                     default=r'H:\ntu_paper_data\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons')
    parser.add_argument('--ignored_sample_path',
                        default=r'H:\ntu_paper_data\NTU_RGBD_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default=r'H:\ntu120\xset\T')

    # benchmark = ['xsub', 'xset']
    benchmark = ['xset']
    part = ['val']
    gaps = [0]
    arg = parser.parse_args()
    for data_path in arg.data_path:
        for b in benchmark:
            for index, p in enumerate(part):
                for gap in gaps:
                    out_path = os.path.join(arg.out_folder, str(gap))
                    out_path = os.path.join(out_path, b)
                    print(out_path)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    print(b, p)
                    gendata(
                        data_path,
                        out_path,
                        gap,
                        arg.ignored_sample_path,
                        benchmark=b,
                        part=p,
                        random_flag=False)
