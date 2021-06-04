import argparse
import json
from pathlib import Path
import os

import pandas as pd
import h5py

def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])

def get_n_frames_hdf5(video_path):
    with h5py.File(video_path, 'r') as f:
        video_data = f['video']
        return len(video_data)

def generate_json_from_folder(video_dir_path,dst_json_path):

    dst_data = {}
    class_list=load_classes("mini-kinetics-200-classes.txt")

    dst_data['labels']=class_list
    dst_data['database'] = {}

    for label in sorted(os.listdir(str(video_dir_path))):
    # for label in class_list:
        print(label)
        for clip_name_path in os.listdir(os.path.join(str(video_dir_path),label)):
            if clip_name_path[-5:]==".hdf5":
                try:
                    clip_name=clip_name_path[:-5]
                    dst_data['database'][clip_name]={"subset":"validation"}

                    video_path = video_dir_path / label / clip_name_path
                    if video_path.exists():

                        n_frames = get_n_frames_hdf5(video_path)
                        # print(n_frames)
                        dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 1)}
                        # dst_data['database'][clip_name]['annotations']={'label':label}
                except:
                    pass

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        basename = '%s_%s_%s' % (row['youtube_id'], '%06d' % row['time_start'],
                                 '%06d' % row['time_end'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database

def load_classes(classes_txt_path):

    class_list=[]
    f=open(classes_txt_path,'r')

    for classname in f.readlines():
        class_list.append(classname.strip("\n"))
    return class_list

def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg or hdf5).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('video_type',
                        default='jpg',
                        type=str,
                        help=('jpg or hdf5'))
    parser.add_argument('dst_path',
                        default=None,
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    assert args.video_type in ['jpg', 'hdf5']

    generate_json_from_folder(args.video_path, args.dst_path)