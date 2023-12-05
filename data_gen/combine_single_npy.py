from tqdm import tqdm
import os
import numpy as np
import pickle

max_frame = 60
num_joint = 25
max_body_true = 2


def read_npy(input_path):
    try:
        data = np.load(input_path, allow_pickle=True).item()
    except:
        data = np.load(input_path, allow_pickle=True)
    return data


def save_npy(npy_save_path, file_name, npy_data):
    # if not (os.path.exists(npy_save_path)):
    npy_save_path = os.path.join(npy_save_path, file_name)
    np.save(npy_save_path, npy_data)


def read_pkl(label_path):
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    return sample_name, label


def read_pkl(label_path):
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    return sample_name, label


def combine_npy(input_dir, out_path, num_fenzhi, part):
    data = None
    label_list = []
    sample_name_list = []
    for i in range(num_fenzhi):
        if i == 0:
            data = read_npy(os.path.join(input_dir, '{}_data_joint_{}.npy'.format(part, str(i))))
        else:
            data = np.concatenate(
                [data, read_npy(os.path.join(input_dir, '{}_data_joint_{}.npy'.format(part, str(i))))], axis=0)
            # data = np.concatenate([data, temp_data],axis=0)

        sample_name, label = read_pkl(os.path.join(input_dir, '{}_label_{}.pkl'.format(part, str(i))))
        label_list = label_list + label
        sample_name_list = sample_name_list + sample_name
    save_npy(out_path, '{}_data_joint'.format(part), data)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name_list, label_list), f)


def combine_for_inference(input_dir, out_path, part, num_fenzhi=1):  # 专门用来比较sota用的
    sample_name = []
    sample_label = []
    for root, dirs, files in os.walk(input_dir):
        files_len = len(files)
        files_num = files_len // num_fenzhi
        fp = np.zeros((files_num, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
        for i in range(num_fenzhi):
            for index, file_index in enumerate(tqdm(range(i * files_num, (i + 1) * files_num))):
                file = files[file_index]
                label = int(file[file.find('A') + 1:file.find('A') + 4]) - 1
                data = read_npy(os.path.join(root, file))
                # print(data.shape)

                fp[index, :, :, :, :] = data[0, :, :, :, :]
                sample_label.append(label)
                sample_name.append(file)
            with open('{}/{}_label_{}.pkl'.format(out_path, part, i), 'wb') as f:
                pickle.dump((sample_name, list(sample_label)), f)
            np.save('{}/{}_data_joint_{}.npy'.format(out_path, part, i), fp)
            sample_name.clear()
            sample_label.clear()


def combine(input_dir, out_path, part):
    sample_name = []
    sample_label = []
    for root, dirs, files in os.walk(input_dir):
        files_len = len(files)
        fp = np.zeros((files_len, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
        for index, file in enumerate(tqdm(files)):
            label = int(file[file.find('A') + 1:file.find('A') + 4]) - 1
            data = read_npy(os.path.join(root, file))
            fp[index, :, :, :, :, :] = data[0, :, :, :, :, :]
            sample_label.append(label)
            sample_name.append(file)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def combine_dict(input_dir, out_path, part):
    sample_name = []
    sample_label = []
    for root, dirs, files in os.walk(input_dir):
        files_len = len(files)
        fp = np.zeros((files_len, 3, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
        # fp = [[np.zeros((2, 3, max_frame, num_joint, max_body_true), dtype=np.float32),
        #        np.zeros((1, 3, 1, num_joint, max_body_true), dtype=np.float32)]] * files_len
        for index, file in enumerate(tqdm(files)):
            label = int(file[file.find('A') + 1:file.find('A') + 4]) - 1
            data = read_npy(os.path.join(root, file))
            fp[index, 0:2, :, :, :, :] = data['data'][:, :, :, :, :]
            fp[index, 2:, :, 0:1, :, :] = data['last_frame'][:, :, :, :, :]
            sample_label.append(label)
            sample_name.append(file)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


def split_npy(data_path, label_path, out_path, num_fenzhi, part):
    data = read_npy(data_path)
    sample_name, sample_label = read_pkl(label_path)
    length = len(sample_label)
    fenzhi_len = length // num_fenzhi
    for i in range(num_fenzhi):
        with open('{}/{}_label_{}.pkl'.format(out_path, part, i), 'wb') as f:
            pickle.dump(
                (sample_name[i * fenzhi_len:(i + 1) * fenzhi_len], sample_label[i * fenzhi_len:(i + 1) * fenzhi_len]),
                f)
        np.save('{}/{}_data_joint_{}.npy'.format(out_path, part, i), data[i * fenzhi_len:(i + 1) * fenzhi_len])
def print_pkl(label_path):
    sample_name, label=read_pkl(label_path)
    print(label)

if __name__ == '__main__':
    # combine_npy(input_dir=r'H:\ntu120\xsub\0-32',
    #             out_path=r'H:\ntu120\xsub',
    #             num_fenzhi=2, part='val')
    # combine_for_inference(input_dir=r'H:\ntu120\xset\T',
    #                       out_path=r'H:\ntu120\xset', part='val')
    # combine_dict(input_dir=r'H:\ntu120\xset\v1\train',
    #              out_path=r'H:\ntu120\xset\v1', part='train')
    split_npy(data_path=r'I:\ntu120\set\T\4\val_data_joint.npy',
              label_path=r'I:\ntu120\set\T\4\val_label.pkl',
              out_path=r'I:\ntu120\set\T\4', num_fenzhi=2,
              part='val')  # 把数据分割开，专门给effenicent和TSGCNeXt-main

    # print_pkl(label_path=r'I:\ntu120\sub\Tabalation\val_label_4.pkl')
