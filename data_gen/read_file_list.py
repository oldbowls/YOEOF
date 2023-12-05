import pickle
from tqdm import tqdm
import os


def gen_list(data_path, out_path, part):
    sample_name_list = []
    sample_start_index = []
    filelists = sorted(os.listdir(data_path))
    for filename in tqdm(filelists):
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        # if action_class - 1 not in training_class:
        #     continue
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        start_index = filename.split(".")[1].split("_")[-1]
        # print(start_index)
        sample_start_index.append(start_index)
        sample_name = filename.split(".")[0].split("_")[-1] + ".skeleton"
        # print(sample_name)
        sample_name_list.append(sample_name)
    with open('{}/{}_index.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name_list, list(sample_start_index)), f)


if __name__ == '__main__':
    parts = ['train', 'val']
    data_path = [r'H:\ntu_paper_data\ntu\v3 0-59 dis\1\train', r'H:\ntu_paper_data\ntu\v3 0-59 dis\1\test']
    out_path = r'H:\ntu_paper_data\ntu\v3 0-59 dis\1'
    for index, part in enumerate(parts):
        gen_list(data_path=data_path[index], out_path=out_path, part=part)
