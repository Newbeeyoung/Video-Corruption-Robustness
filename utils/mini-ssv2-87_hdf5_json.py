import argparse
import json
from pathlib import Path
import os

import pandas as pd

from utils import get_n_frames, get_n_frames_hdf5

def generate_json_from_folder(video_dir_path,dst_json_path):

    dst_data = {}
    # dst_data['labels'] = os.listdir(video_dir_path)
    # labels = load_labels(train_csv_path)
    # dst_data['labels'] = [label.replace(" ","_") for label in labels]
    class_list=load_classes("mini-ssv2-87-classes.txt")

    dst_data['labels']=class_list
    dst_data['database'] = {}

    # for label in os.listdir(video_dir_path):
    for label in class_list:
        print(label)
        for clip_name_path in os.listdir(os.path.join(str(video_dir_path),label)):
            clip_name=clip_name_path[:-5]
            dst_data['database'][clip_name]={"subset":"validation"}

            video_path = video_dir_path / label / clip_name_path
            if video_path.exists():

                n_frames = get_n_frames_hdf5(video_path)
                # print(n_frames)
                dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 1)}
                # dst_data['database'][clip_name]['annotations']={'label':label}

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


def load_classes(classes_txt_path):

    class_list=[]
    f=open(classes_txt_path,'r')

    for classname in f.readlines():
        class_list.append(classname.strip("\n"))
    return class_list


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