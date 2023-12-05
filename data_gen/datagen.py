max_frame = 18  #
num_joint = 11
data_dim = 2
fenge_num = 18  # 按照多少帧进行分割，于max_frame作用相似

import sys
import numpy as np
import os
import json
from data_gen.mypreprocess import *

from tqdm import tqdm


# import warnings
# warnings.filterwarnings('error')


def load_json(file_path):
    with open(file_path, 'r') as data_file:
        data = json.load(data_file)
    return data


def write_json(file_path, fp, file_name):
    with open(os.path.join(file_path, file_name + '.json'), 'w') as data_file:
        data_file.write(json.dumps(fp))


# def data_gen(data_path, out_path):
#     # files = os.listdir(data_path)
#     same = 0
#     unsame = 0
#     for root, dirs, files in os.walk(data_path):
#         process = tqdm(files)
#         for i, s in enumerate(process):
#             data = load_json(os.path.join(root, s))
#             data['data'] = np.array(data['data'])  # list转numpy
#             fp = np.zeros((max_frame, num_joint, data_dim)
#                           , dtype=np.float32)
#
#             # if int(data['num_frame']) == data['data'].shape[0]:
#             #     same += 1
#             #     print('collect', os.path.join(root, s))
#             #
#             # else:
#             #     print(os.path.join(root, s))
#             #     unsame += 1
#             #     print(unsame)
#             if data['num_frame'] < max_frame:  # 小于最大帧数不进行分割而是需要填充
#                 fp[0:data['data'].shape[0], :, :] = data['data']  # 赋值并转换为numpy
#                 # fp[0:data['data'].shape[0], 0:num_joint, 0:data_dim] = data['data']  # 赋值并转换为numpy
#                 fp = pre_normalization(fp, max_frame, data['num_frame'], False, max_frame)
#             # print(type(fp), '<')
#             elif data['num_frame'] > max_frame:  # 大于最大帧数会进行分割进行
#                 fp = pre_normalization(data=data['data'], max_frame=max_frame, frame_num=data['num_frame'],
#                                        fenge_num=max_frame, class_id=int(data['label']), flag=True)
#             # print(type(fp), '>')
#             else:
#                 fp = data['data']
#
#             if type(fp) is np.ndarray:  # 如果不进行数据分割或者数据填充
#                 data['data'] = fp.tolist()
#                 data['num_frame'] = max_frame
#                 write_json(out_path, data, s.split('.')[0])
#                 # if len(data['data']) == 0:
#                 # print('np is em', os.path.join(root, s))
#             elif isinstance(fp, list):
#                 for index, i in enumerate(fp):
#                     data['data'] = i
#                     data['num_frame'] = max_frame
#                     data['data'] = data['data'].tolist()
#                     # if len(data['data']) == 0:
#                     #     print('list is em', os.path.join(root, s))
#                     write_json(out_path, data, s.split('.')[0] + '_' + str(index))


def frame_index(num, file_flag, data):  # num表示再原来样本中的第几个样本,file_flag表示是否为原文件
    if file_flag == 'init':
        data['start_frame'] = num
        data['end_frame'] = num + max_frame
    else:
        data['start_frame'] = num
        data['end_frame'] = num + max_frame
    return data


def data_gen(data_path, out_path):
    for root, dirs, files in os.walk(data_path):
        process = tqdm(files)
        for i, file in enumerate(process):
            data = load_json(os.path.join(root, file))
            data['data'] = np.array(data['data'])  # list转numpy
            print(os.path.join(root, file))
            fp = np.zeros((max_frame, num_joint, data_dim)
                          , dtype=np.float32)
            if data['num_frame'] < max_frame:  # 小于最大帧数不进行分割而是需要填充
                fp[0:data['data'].shape[0], :, :] = data['data']  # 赋值并转换为numpy
                fp = frame_padding(int(data['num_frame']), max_frame, fp)
            elif data['num_frame'] > max_frame:  # 大于最大帧数会进行分割进行
                fp = fenge(data['data'], fenge_num, int(data['label']))
            else:  # 等于最大帧数刚好直接赋值
                fp = data['data']
            if type(fp) is np.ndarray:  # 如果不进行数据分割,那么fp就是np.ndarray类型的数据
                fp = pre_normalization(fp)
                if fp is not None:
                    data['data'] = fp.tolist()
                    data['num_frame'] = max_frame
                    data['init_file'] = file
                    frame_index(0, 'init', data)
                    write_json(out_path, data, data['label'] + '_' + file.split('.')[0])
                else:
                    print('None')
            elif isinstance(fp, list):  # 如果进行数据分割,那么fp就是list[np.ndarray,np.ndarray...]类型的数据
                for index, i in enumerate(fp):
                    temp = pre_normalization(i)
                    if temp is not None:
                        data['data'] = temp
                        data['num_frame'] = max_frame
                        data['data'] = data['data'].tolist()
                        data['init_file'] = file
                        data = frame_index(index, '', data)
                        # if len(data['data']) == 0:
                        #     print('list is em', os.path.join(root, s))
                        write_json(out_path, data, data['label'] + '_' + file.split('.')[0] + '_' + str(index))
                    else:
                        print('None')
            else:
                print('fuck')


def precess_pipeline(input_path, out_path):
    for root, dirs, files in os.walk(input_path):
        process = tqdm(files)

        for i, file in enumerate(process):
            data = load_json(os.path.join(root, file))
            # print(os.path.join(root, file))
            data['data'] = np.array(data['data'])
            fp = np.zeros((max_frame, num_joint, data_dim)
                          , dtype=np.float32)
            fp = up_or_down_sample(data, fp, max_frame, fenge_num)  # 对过短或者过长的样本近行分割或者填充
            if type(fp) is np.ndarray:  # 如果不进行数据分割,那么fp就是np.ndarray类型的数据
                '''
                pipeline
                '''
                fp = center_coordinate(fp)  # 统一减去中心节点坐标，区间缩放可能就可以将数据进行中心节点化，但是个人觉得这一步还是需要滴
                fp = my_normalize_bone(fp, os.path.join(root, file))
                # fp = my_normalize_sample(fp)
                if fp is not None:
                    data['data'] = fp.tolist()
                    data['num_frame'] = max_frame
                    data['init_file'] = file
                    frame_index(0, 'init', data)
                    write_json(out_path, data, str(data['label']) + '_' + file.split('.')[0])
                else:
                    print('None')
            elif isinstance(fp, list):  # 如果进行数据分割,那么fp就是list[np.ndarray,np.ndarray...]类型的数据
                for index, i_data in enumerate(fp):
                    fp = center_coordinate(i_data)
                    fp = my_normalize_bone(fp, os.path.join(root, file))
                    # fp = my_normalize_sample(fp)
                    if fp is not None:
                        data['data'] = fp.tolist()
                        data['num_frame'] = max_frame
                        data['init_file'] = file
                        data = frame_index(0, 'init', data)
                        write_json(out_path, data, str(data['label']) + '_' + str(file.split('.')[0]) + '_' + str(index))
                    else:
                        print('None')
            else:
                print('fuck')


precess_pipeline(r'G:\动作训练最终数据\第六次更改\active sample',
                 r'G:\动作训练最终数据\第六次更改\pre 18')
