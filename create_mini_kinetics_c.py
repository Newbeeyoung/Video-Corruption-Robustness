import torch
import h5py
import os
import numpy as np
import cv2
import skimage as sk
from skimage.color import hsv2rgb,rgb2hsv
import subprocess
import json
from pathlib import Path
import torchvision.transforms as transforms

from dataloader import VideoDatasetHdf5
from utils import VideoLoaderHDF5,ImageToHdf5
from video_transforms import *
from create_mini_ssv2_c import *
# Corruptions

#Single Image

#Image List

def motion_blur(x,severity):

    motion_overlapping_frames=[3,5,7,9,11]

    c=motion_overlapping_frames[severity-1]

    clip=np.asarray(x)
    blur_clip=[]
    for i in range(c,clip.shape[0]-c):
        blur_image=np.sum(clip[i-c:i+c],axis=0,dtype=np.float)/(2.0*c)
        blur_clip.append(np.array(blur_image,dtype=np.uint8))
    return blur_clip


def frame_rate(src,dst,severity):

    c=[20,16,12,9,6][severity-1]
    return_code = subprocess.call(
        ["ffmpeg","-y",  "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-fps", str(c), dst])

    return return_code
#Make Corruption Data

if __name__=="__main__":
    # Single Image & Image List
    val_hdf5_video_path=Path("/data/Dataset/kinetics/val_hdf5")
    val_video_path=Path("/data/Dataset/kinetics/val")
    val_c_video_dir="/data/Dataset/kinetics/mini_kinetics200-c"
    val_annotation_path = Path("./data/mini-kinetics-200_val.json")

    hdf5_path_formatter = (lambda root_path, label, video_id: root_path /
                                                               label / '{}.hdf5'.format(video_id))
    video_path_formatter= (lambda root_path, label, video_id: root_path /
                                                               label / '{}.webm'.format(video_id))
    video_c_path_formatter= (lambda root_path, label, video_id: root_path /
                                                               label / '{}.mp4'.format(video_id))

    val_hdf5_dataset = VideoDatasetHdf5(val_hdf5_video_path,
                                   val_annotation_path,
                                   'validation',
                                   target_transform=None,
                                   video_loader=VideoLoaderHDF5(),
                                   video_path_formatter=hdf5_path_formatter)

    from torch.utils.data.dataloader import default_collate

    def id_collate(batch):
        new_batch = []
        labels=[]
        paths = []
        for _batch in batch:
            new_batch.append(_batch[0])
            labels.append(_batch[1])
            paths.append(_batch[-1])
        return new_batch,labels,paths

    val_hdf5_dataloader= torch.utils.data.DataLoader(val_hdf5_dataset, batch_size=1, shuffle=False, num_workers=8,pin_memory=True,collate_fn=id_collate)

    scale_size=256
    input_size=224

    img_process = []

    # img_process.append(transforms.ToPILImage())
    img_process.append(transforms.Resize(scale_size))
    img_process.append(transforms.CenterCrop(input_size))

    img_process=transforms.Compose(img_process)
    #
    SingleImageCorruption={"shot_noise":shot_noise,"rain":rain,"contrast":contrast,"brightness":brightness,"saturate":saturate}

    error_txt=open("error.txt","a")
    # import pdb
    #
    for key in SingleImageCorruption.keys():
        single_corruption = SingleImageCorruption[key]

        if not os.path.exists(os.path.join(val_c_video_dir, key)):
            os.mkdir(os.path.join(val_c_video_dir, key))

        for severity in range(1, 6):
            val_c_video_path = "{}/{}/{}".format(val_c_video_dir, key, severity)
            if not os.path.exists(val_c_video_path):
                os.mkdir(val_c_video_path)
            print(key,severity)
            for i,(clip,label,video_id) in enumerate(val_hdf5_dataloader):

                for n in range(len(clip)):
                    corrupted_clip=[]

                    if not os.path.exists(os.path.join(val_c_video_path, label[n])):
                        os.mkdir(os.path.join(val_c_video_path, label[n]))

                    for image in clip[n]:
                        corrupted_image=np.array(single_corruption(img_process(image),severity),dtype=np.uint8)
                        corrupted_clip.append(corrupted_image)

                    hdf5_path=hdf5_path_formatter(Path(val_c_video_path),label[n],video_id[n])
                    # hdf5_path=os.path.join(val_c_video_path,label[n],video_id[n])
                    success_code=ImageToHdf5(corrupted_clip,str(hdf5_path))
                    if success_code==0:
                        error_txt.write(hdf5_path)
                        error_txt.write("\n")
    #

    clip_process = []

    clip_process += [
        GroupScale(scale_size),
        GroupCenterCrop(input_size),
        Stack(threed_data=True)
    ]

    clip_process = transforms.Compose(clip_process)

    ClipCorruption={"motion_blur":motion_blur,"fog":fog}

    for key in ClipCorruption.keys():
        clip_corruption = ClipCorruption[key]

        if not os.path.exists(os.path.join(val_c_video_dir, key)):
            os.mkdir(os.path.join(val_c_video_dir, key))

        for severity in range(1, 6):
            val_c_video_path = "{}/{}/{}".format(val_c_video_dir, key, severity)
            if not os.path.exists(val_c_video_path):
                os.mkdir(val_c_video_path)
                print(key,severity)

            for i, (clip,label,video_id) in enumerate(val_hdf5_dataloader):

                for n in range(len(clip)):

                    if not os.path.exists(os.path.join(val_c_video_path, label[n])):
                        os.mkdir(os.path.join(val_c_video_path, label[n]))

                    corrupted_clip = clip_corruption(clip_process(clip[n]), severity)

                    hdf5_path = hdf5_path_formatter(Path(val_c_video_path), label[n], video_id[n])
                    success_code = ImageToHdf5(corrupted_clip, str(hdf5_path))
                    if success_code==0:
                        error_txt.write(hdf5_path)
                        error_txt.write("\n")

    FfmpegCorruption={"bit_error":bit_error,"h265_crf":h265_crf,"h265_abr":h265_abr,"h264_crf":h264_crf,"h264_abr":h264_abr,"frame_rate":frame_rate}

    for key in FfmpegCorruption.keys():

        if not os.path.exists(os.path.join(val_c_video_dir, key)):
            os.mkdir(os.path.join(val_c_video_dir, key))

        for severity in range(1, 6):
            val_c_video_path = "{}/{}/{}".format(val_c_video_dir, key, severity)
            if not os.path.exists(val_c_video_path):
                os.mkdir(val_c_video_path)

            for i,(clip,label,video_id) in enumerate(val_hdf5_dataloader):
                for n in range(len(clip)):

                    if not os.path.exists(os.path.join(val_c_video_path, label[n])):
                        os.mkdir(os.path.join(val_c_video_path, label[n]))

                    video_corruption=FfmpegCorruption[key]

                    src= video_path_formatter(val_video_path,label[n],video_id[n])
                    dst= video_c_path_formatter(Path(val_c_video_path),label[n],video_id[n])

                    success_code=video_corruption(src,dst,severity)
                    if success_code!=0:
                        error_txt.write(str(dst))
                        error_txt.write("\n")




