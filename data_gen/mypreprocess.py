import numpy as np

np.seterr(all='raise')

dynamic_action = [0, 1, 2, 3, 4, 5]  # 动态且循环动作编号
dynamic_action_not_cricle_action = [6]  # 动态且不循环动作编号


# bone_connect = []


# def pre_normalization(data, max_frame, frame_num, fenge_num, class_id=None, center_index=9, flag=False,
#
#                       xaxis=[1, 2]):  # 填充帧数,坐标数据归一化，与x轴平行
#
#     """
#         与x轴平行
#     """
#     joint_rshoulder = data[0, xaxis[0], :]  # 节点0的x,y坐标
#     joint_lshoulder = data[0, xaxis[1], :]  # 节点1的x,y坐标
#     main_vector = joint_lshoulder - joint_rshoulder
#     costheta = np.dot(main_vector, np.array([1, 0])) / (np.linalg.norm(main_vector))  # 求与x轴的夹角的余弦值
#     sintheta = (1 - costheta * costheta) ** 0.5
#     rotation_matrix = np.array([[costheta, sintheta], [-sintheta, costheta]])  # 二维矩阵的givens旋转矩阵
#     data = np.dot(data, rotation_matrix)
#
#     '''`
#         坐标数据归一化
#     '''
#     main_center = data[0, center_index, :]
#     data = data - main_center
#
#
#     return data

def pre_normalization(data, center_index=9, xaxis=[1, 2]):  # 填充帧数,坐标数据归一化，与x轴平行

    """
        与x轴平行
    """
    joint_rshoulder = data[0, xaxis[0], :]  # 节点0的x,y坐标
    joint_lshoulder = data[0, xaxis[1], :]  # 节点1的x,y坐标
    main_vector = joint_lshoulder - joint_rshoulder
    fenmu = np.linalg.norm(main_vector)
    if fenmu:
        costheta = np.dot(main_vector, np.array([1, 0])) / (np.linalg.norm(main_vector))  # 求与x轴的夹角的余弦值
    else:
        return None
    sintheta = (1 - costheta * costheta) ** 0.5
    rotation_matrix = np.array([[costheta, sintheta], [-sintheta, costheta]])  # 二维矩阵的givens旋转矩阵
    data = np.dot(data, rotation_matrix)

    '''`
        坐标数据减去中心节点
    '''
    main_center = data[0, center_index, :]
    data = data - main_center

    return data
    # print(pre_normalization(np.array([[[3, 4], [5, 6]]],dtype=np.float32), 10, 1,[0,1],0))
    #
    # def sampling(data,sampleRate):#进行帧采样,samplerate隔几帧进行一次采样
    #     for i in range(sampleRate):
    #         for


def parallel(data, xaxis=[1, 2]):
    """
        与x轴平行
    """
    joint_rshoulder = data[0, xaxis[0], :]  # 节点0的x,y坐标
    joint_lshoulder = data[0, xaxis[1], :]  # 节点1的x,y坐标
    main_vector = joint_lshoulder - joint_rshoulder
    fenmu = np.linalg.norm(main_vector)
    if fenmu:
        costheta = np.dot(main_vector, np.array([1, 0])) / (np.linalg.norm(main_vector))  # 求与x轴的夹角的余弦值
    else:
        return None
    sintheta = (1 - costheta * costheta) ** 0.5
    rotation_matrix = np.array([[costheta, sintheta], [-sintheta, costheta]])  # 二维矩阵的givens旋转矩阵
    data = np.dot(data, rotation_matrix)

    return data


def center_coordinate(data, center_index=9):
    main_center = data[0, center_index, :]  # data为一个动作，选取第0帧，第九个节点的的坐标值
    data = data - main_center
    return data


# def frame_padding(frame_num, max_frame, data, label):
#     if frame_num < max_frame and (label not in dynamic_action) and (
#             label not in dynamic_action_not_cricle_action):  # 可以对静态动作进行填充
#         length = max_frame - frame_num
#         num = length // frame_num
#         end_index = frame_num
#         for i in range(num):
#             start_index = frame_num * (i + 1)
#             end_index = start_index + frame_num
#             data[start_index:end_index, :, :] = data[0:frame_num, :, :]
#         length = max_frame % frame_num
#         data[end_index:max_frame, :, :] = data[0:length, :, :]
#     elif frame_num > max_frame:
#         print('error:frame>max_frame')
#         return
#     elif frame_num == max_frame:
#         pass
#     return data
def frame_padding(frame_num, max_frame, data, label):
    if frame_num < max_frame and (label not in dynamic_action) and (
            label not in dynamic_action_not_cricle_action):  # 可以对静态动作进行填充
        length = max_frame - frame_num
        num = length // frame_num
        end_index = frame_num
        for i in range(num):
            start_index = frame_num * (i + 1)
            end_index = start_index + frame_num
            data[start_index:end_index, :, :] = data[0:frame_num, :, :]
        length = max_frame % frame_num
        data[end_index:max_frame, :, :] = data[0:length, :, :]
    else:
        return None
    return data


def up_or_down_sample(data, fp, max_frame, fenge_num):
    if data['num_frame'] < max_frame:  # 小于最大帧数不进行分割而是需要填充up
        fp[0:data['data'].shape[0], :, :] = data['data']  # 赋值并转换为numpy
        fp = frame_padding(int(data['num_frame']), max_frame, fp, int(data['label']))
        return None
    elif data['num_frame'] > max_frame:  # 大于最大帧数会进行分割进行
        fp = fenge(data['data'], fenge_num, int(data['label']))
    else:  # 等于最大帧数刚好直接赋值
        fp = data['data']
    return fp


def fenge(data, frame_num, class_id):  # frame_num为分割的帧数
    dim0 = data.shape[0]
    # print(dim0)
    datalist = []
    if dim0 < frame_num:
        print('error in fenge function')
        return
    for i in range(0, dim0, 1):
        if (i + frame_num) <= dim0:
            # print(data[i:i + frame_num, :, :])
            datalist.append(data[i:i + frame_num, :, :])
            '''
            考虑要不要加？
            '''
        # elif class_id in dynamic_action:  # 判断是否为循环类动作,如果是的话
        #     temp = np.zeros((frame_num, 11, 2))
        #     temp[:dim0 - i, :, :] = data[i:, :, :]
        #     # print(dim0 - frame_num -i + dim0,frame_num + i - dim0)
        #     temp[dim0 - i:, :, :] = data[0:i + frame_num - dim0, :, :]
        #     datalist.append(temp)
    return datalist


def normalize_bone(video):  # 此步目的是对骨骼大小进行归一化处理，这里的video代表一个完整的动作样本
    """

    :param video:
    :return:
    """
    bone_connect = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (20, 4),
                    (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), \
                    (0, 16), (16, 17), (17, 18), (18, 19), (0, 12), (12, 13), (13, 14), (14, 15)]
    new_video = np.zeros_like(video)
    new_video[:, 0:3] = video[:, 0:3]  # 所有样本的第一个根节点的坐标
    #     dif_norm_max = 0.
    #     dif_norm_min = 100.
    #     for bone in bone_connect:
    #         tmp_max = np.amax(np.linalg.norm(video[:,bone[1]*3:bone[1]*3+3]-video[:,bone[0]*3:bone[0]*3+3], axis=1))
    #         tmp_min = np.amin(np.linalg.norm(video[:,bone[1]*3:bone[1]*3+3]-video[:,bone[0]*3:bone[0]*3+3], axis=1))
    #         if tmp_max > dif_norm_max:
    #             dif_norm_max = tmp_max
    #         if tmp_min < dif_norm_min:
    #             dif_norm_min = tmp_min
    for bone in bone_connect:  # 从根节点开始选取每一个坐标的坐标差
        dif = video[:, bone[1] * 3:bone[1] * 3 + 3] - video[:, bone[0] * 3:bone[0] * 3 + 3]  # 求坐标差
        dif_norm = np.linalg.norm(dif, axis=1)  # 求二范数、模
        new_video[:, bone[1] * 3:bone[1] * 3 + 3] = new_video[:, bone[0] * 3:bone[0] * 3 + 3] + dif / np.expand_dims(
            dif_norm, axis=1)  # * np.expand_dims((dif_norm / dif_norm_max),
        # axis=1)，以（0，1）为例，以0为根节点，求出（0，1）的单位向量，并给0节点加上单位向量。最终在这个循环完成之后形成以0为根节点，各个骨架长度为1的的一整副骨架
    return new_video


def normalize_video(video):  # 这里的video代表一个完整的动作样本,此函数最终会将样本大小区间缩放到（-1，1）
    """

    :param video:
    :return:
    """
    max_75 = np.amax(video, axis=0)  # 寻找一个完整动作样本种每一帧的最大和最小的元素，axis=0 代表列, axis=1 代表行
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0, 75, 3)])  # [max_75[i] for i in range(0, 75, 3)],将所有的x点提取出来
    max_y = np.max([max_75[i] for i in range(1, 75, 3)])
    max_z = np.max([max_75[i] for i in range(2, 75, 3)])
    min_x = np.min([min_75[i] for i in range(0, 75, 3)])
    min_y = np.min([min_75[i] for i in range(1, 75, 3)])
    min_z = np.min([min_75[i] for i in range(2, 75, 3)])
    # 以上为找到最大和最小的x,y,z值，注意x,y,z不一定都老总与同一个节点
    norm = np.zeros_like(video)
    for i in range(0, 75, 3):  # 步长为3，作者已经将一帧中所有坐标放至一个向量中
        norm[:, i] = 2 * (video[:, i] - min_x) / (max_x - min_x) - 1  # (max_x - min_x)为最大和最小x的距离（是一个标量）
        norm[:, i + 1] = 2 * (video[:, i + 1] - min_y) / (max_y - min_y) - 1
        norm[:, i + 2] = 2 * (video[:, i + 2] - min_z) / (max_z - min_z) - 1
        # 此步核心思想为区间缩放，平移；(video[:, i] - min_x) / (max_x - min_x)-》（0，1）；（*2）-》（0，2）；（*2-1）-》（-1，1）；nice！
    return norm






def my_normalize_sample(sample):  # 这里的sample代表一个完整的动作样本,此函数最终会将样本大小区间缩放到（-1，1)
    """

    Args:
        sample: 这里的sample代表一个完整的动作样本,(帧数，节点数，节点特征数)

    Returns:

    """
    xs = sample[:, :, 0].flatten()  # 提取出所有的x坐标值,flatten()为压缩为1维度的
    ys = sample[:, :, 1].flatten()  # 提取出所有的x坐标值，flatten()为压缩为1维度的
    max_x = np.max(xs)
    max_y = np.max(ys)
    min_x = np.min(xs)
    min_y = np.min(ys)

    norm_sample = np.zeros_like(sample)

    norm_sample[:, :, 0] = 2 * (sample[:, :, 0] - min_x) / (max_x - min_x) - 1
    norm_sample[:, :, 1] = 2 * (sample[:, :, 1] - min_y) / (max_y - min_y) - 1

    return norm_sample
