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
# Corruptions

#Single Image
def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


    #Rain
def generate_random_lines(imshape, slant, drop_length, rain_type):
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    # if rain_type.lower()=='drizzle':

    if rain_type == 1:
        no_of_drops = area // 770
        drop_length = 10
        # print("drizzle")
    # elif rain_type.lower()=='heavy':
    elif rain_type == 2:
        no_of_drops = area // 770
        drop_length = 30
    # elif rain_type.lower()=='torrential':
    elif rain_type == 3:
        no_of_drops = area // 770
        drop_length = 60
        # print("heavy")
    elif rain_type == 4:
        no_of_drops = area // 500
        drop_length = 60
    elif rain_type == 5:
        no_of_drops = area // 400
        drop_length = 80
        # print('torrential')

    for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops, drop_length

def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, darken):
    imshape = image.shape
    rain_mask = np.zeros((imshape[0], imshape[1]))
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(rain_mask, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                 drop_color, drop_width)

    rain_mask = np.stack((rain_mask, rain_mask, rain_mask), axis=2)
    image_rain = image + np.array(rain_mask * (1 - image / 255.0) * (1 - np.mean(image) / 255.0), dtype=np.uint8)
    blur_rain = cv2.blur(image_rain, (3, 3))  ## rainy view are blurry
    image_RGB = np.array(blur_rain * rain_mask / 255.0 + image * (1 - rain_mask / 255.0))
    # blur_rain_mask=rain_mask
    image_RGB = np.array(image_RGB) / 255.
    means = np.mean(image_RGB, axis=(0, 1), keepdims=True)
    image_RGB = np.array(np.clip((image_RGB - means) * darken + means, 0, 1) * 255, dtype=np.uint8)

    return image_RGB

##rain_type='drizzle','heavy','torrential'
def rain(image, severity=2):  ## (200,200,200) a shade of gray
    # verify_image(image)
    image = np.asarray(image)
    slant = -1
    drop_length = 20
    drop_width = 1
    drop_color = (220, 220, 220)
    rain_type = severity
    darken_coefficient = [0.8, 0.8, 0.7, 0.6, 0.5]
    slant_extreme = slant

    imshape = image.shape
    if slant_extreme == -1:
        slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
    output = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                          darken_coefficient[severity - 1])
    image_RGB = output

    return image_RGB

def contrast(x, severity=1):
    # c = [0.4, .3, .2, .1, .05][severity - 1]
    c = [0.5, 0.4, .3, .2, .1][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

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


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def fog(x,severity):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    fog_layer = c[0] * plasma_fractal(wibbledecay=c[1])
    fog_clip=[]
    for i,image in enumerate(x):
        image = np.array(image) / 255.
        max_val = image.max()
        width,height,depth=image.shape
        image=image[width//2-112:width//2+112,height//2-112:height//2+112]
        image += np.repeat(fog_layer[:224, :224][..., np.newaxis],3,axis=2)
        fog_image = np.array(np.clip(image * max_val / (max_val + c[0]), 0, 1) * 255,dtype=np.uint8)
        fog_clip.append(fog_image)

    return fog_clip

# Whole Video
def bit_error(src,dst,severity):
    c=[100000, 50000, 30000, 20000, 10000][severity-1]
    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", src, "-vcodec", "libx265", "-c", "copy", "-bsf", "noise={}".format(str(c)),
         dst])

    return return_code

def h264_crf(src,dst,severity):
    c=[23,30,37,44,51][severity-1]
    return_code = subprocess.call(
        ["ffmpeg", "-i", src, "-vcodec", "libx264", "-crf", str(c), "-fps", "24", dst])

    return return_code

def h264_abr(src,dst,severity):

    c=[2,4,8,16,32][severity-1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    return_code = subprocess.call(
        ["ffmpeg","-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
         bit_rate, dst])

    return return_code

def h265_crf(src, dst, severity):
    c = [27, 33, 39, 45, 51][severity - 1]
    return_code = subprocess.call(
        ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'","-vcodec", "libx265", "-crf", str(c), dst])

    return return_code

def h265_abr(src, dst, severity):
    c = [2, 4, 8, 16, 32][severity - 1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    return_code = subprocess.call(
        ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
         bit_rate, dst])

    return return_code

def frame_rate(src,dst,severity):

    c=[10,8,6,4,2][severity-1]
    return_code = subprocess.call(
        ["ffmpeg","-y",  "-i", src, "-vcodec", "libx265", "-fps", str(c), dst])

    return return_code
#Make Corruption Data

# Single Image & Image List
val_hdf5_video_path=Path("/data/Dataset/Something-something-Dataset-V2/val_hdf5")
val_video_path=Path("/data/Dataset/Something-something-Dataset-V2/original/val")
val_c_video_dir="/data/Dataset/Something-something-Dataset-V2/mini_ssv287-c"
val_annotation_path = Path("./data/mini-ssv2-87_val.json")

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

val_hdf5_dataloader= torch.utils.data.DataLoader(val_hdf5_dataset, batch_size=1, shuffle=False, num_workers=1,pin_memory=True,collate_fn=id_collate)

scale_size=224
input_size=224

img_process = []

# img_process.append(transforms.ToPILImage())
img_process.append(transforms.Resize(scale_size))
img_process.append(transforms.CenterCrop(input_size))

img_process=transforms.Compose(img_process)

SingleImageCorruption={"shot_noise":shot_noise,"rain":rain,"contrast":contrast,"brightness":brightness,"saturate":saturate}

error_txt=open("error.txt","w")
import pdb
#
for key in SingleImageCorruption.keys():
    single_corruption = SingleImageCorruption[key]
    val_c_video_path = "/data/Dataset/Something-something-Dataset-V2/mini_ssv287-c/{}".format(key)

    if not os.path.exists(val_c_video_path):
        os.mkdir(val_c_video_path)
    for severity in range(1,6):
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
# clip_process = []
#
# clip_process += [
#     GroupScale(scale_size),
#     GroupCenterCrop(input_size),
#     Stack(threed_data=True)
# ]
#
# clip_process = transforms.Compose(clip_process)
#
# ClipCorruption={"motion_blur":motion_blur,"fog":fog}
# for key in ClipCorruption.keys():
#     clip_corruption = ClipCorruption[key]
#     if not os.path.exists(os.path.join(val_c_video_dir,key)):
#         os.mkdir(os.path.join(val_c_video_dir,key))
#     for severity in range(1,6):
#         val_c_video_path = "{}/{}/{}".format(val_c_video_dir,key,severity)
#         if not os.path.exists(val_c_video_path):
#             os.mkdir(val_c_video_path)
#         print(key,severity)
#
#         for i, (clip,label,video_id) in enumerate(val_hdf5_dataloader):
#
#             for n in range(len(clip)):
#
#                 if not os.path.exists(os.path.join(val_c_video_path, label[n])):
#                     os.mkdir(os.path.join(val_c_video_path, label[n]))
#
#                 corrupted_clip = clip_corruption(clip_process(clip[n]), severity)
#
#                 hdf5_path = hdf5_path_formatter(Path(val_c_video_path), label[n], video_id[n])
#                 success_code = ImageToHdf5(corrupted_clip, str(hdf5_path))
#                 if success_code==0:
#                     error_txt.write(hdf5_path)
#                     error_txt.write("\n")

FfmpegCorruption={"h265_abr":h265_abr,"h264_abr":h264_abr}

for key in FfmpegCorruption.keys():
    if not os.path.exists(os.path.join(val_c_video_dir, key)):
        os.mkdir(os.path.join(val_c_video_dir, key))
    for severity in range(1, 6):
        val_c_video_path = "{}/{}/{}".format(val_c_video_dir, key, severity)
        if not os.path.exists(val_c_video_path):
            os.mkdir(val_c_video_path)
#
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

            break


