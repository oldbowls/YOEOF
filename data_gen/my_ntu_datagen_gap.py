import sys
import random

sys.path.extend(['../'])

import pickle
import argparse

from tqdm import tqdm

from data_gen.preprocess import pre_normalization

# https://arxiv.org/pdf/1604.02808.pdf, Section 3.2
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]

# training_class = [0, 1, 2, 3, 4]#, 5, 6, 7, 8, 9]  # , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

max_body_true = 2
max_body_kinect = 4

num_joint = 25
max_frame = 60

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
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
    index = s.sum(-1).sum(-1) != 0  # select valid frames,s.sum(-1)由最底维度相加，此行代码即为25个骨骼节点和不为0则返回其下标索引。
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels，std计算标准差
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    # max_body表识最大人数，seq_info表识帧数，num_joint表识骨骼节点数，3表识提取x,y,z三个节点的坐标。
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]  # m,n,j是下标，v['x'], v['y'], v['z']是骨关节的x,y,z坐标。
                else:
                    pass

        # select two max energy body，以下三行代码即为选出可靠度最高的max_body个body数据，标准差越小可靠度越高
        energy = np.array([get_nonzero_std(x) for x in data])  # for x in data从高维向低维扩散，x为（帧数，骨骼节点数，节点坐标特征，erergy为x的标准差
    index = energy.argsort()[::-1][0:max_body_true]  # argsort从小到大排序并返回排序后数组的坐标,[::-1]表识步长为1，
    data = data[index]

    data = data.transpose(3, 1, 2, 0)  # 交换数据维度，第一维为骨骼的x,y,z特征，第二位依旧为该系列动作的帧数，第三维依旧为骨骼节点个数，第四维为人数
    return data


def load_data(input_path):
    # data: N C V T M
    try:
        with open(input_path) as f:
            sample_name, sample_start_index = pickle.load(f)
    except:
        # for pickle file from python2
        with open(input_path, 'rb') as f:
            sample_name, sample_start_index = pickle.load(f, encoding='latin1')
    return zip(sample_name, sample_start_index)


def gendata(data_path, out_path, gap, ignored_sample_path=None, benchmark='xview', part='eval', random_flag=False):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in sorted(os.listdir(data_path)):
        if filename in ignored_samples:
            print('continue')
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
        # if action_class-1 not in training_class:
        #     continue

        if benchmark == 'xview':  # 按视角分类数据集,属于camera_id得为训练集
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':  # 按照对象id对数据进行分类
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
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

    # # 输出标签。
    # with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)
    #
    # fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    #
    # # (40091,3,300,25,2)40091表识数据文件的个数，3表示x,y,z特征，300表识该系列动作的最大帧数，num_joint表识骨骼节点数，max_body_true表识最大人数
    # # Fill in the data tensor `fp` one training example a time
    # min = 10000000
    #
    # for i, s in enumerate(tqdm(sample_name)):
    #     data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect,
    #                     num_joint=num_joint)  # data的数据格式为（c,t,v,m),其中t是不定长的,在ntu60中最长是32
    #     if min > data.shape[1]:
    #         min = data.shape[1]
    #         print('min:', min)
    #     fp[i, :, 0:data.shape[1], :, :] = data
    #
    # fp = pre_normalization(fp)
    # np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    # parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--data_path',
                        default=r'H:\ntu_paper_data\nturgbd_skeletons_s018_to_s032')
    # parser.add_argument('--data_path',
    #                     default=r'H:\ntu_paper_data\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons')
    parser.add_argument('--ignored_sample_path',
                        default=r'H:\ntu_paper_data\NTU_RGBD_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default=r'H:\ntu120\sub\Tabalation')

    # benchmark = ['xsub', 'xview']
    benchmark = ['xsub']
    part = ['val']
    gaps = [0]
    arg = parser.parse_args()
    for b in benchmark:
        for index, p in enumerate(part):
            for gap in gaps:
                out_path = os.path.join(arg.out_folder, str(gap))
                out_path = os.path.join(out_path, b)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                print(b, p)
                gendata(
                    arg.data_path,
                    out_path,
                    gap,
                    arg.ignored_sample_path,
                    benchmark=b,
                    part=p,
                    random_flag=False)
