import io

import h5py
from PIL import Image
import os
from pathlib import Path
import numpy as np

class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader
        # self.image_loader=ImageLoaderMx()

    def __call__(self, video_path, frame_indices):
        video = []
        for i in range(len(frame_indices)):
            image_path = video_path / self.image_name_formatter(frame_indices[i])
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):

        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in range(len(frame_indices)):
            # if i < len(video_data):
            #     video.append(Image.open(io.BytesIO(video_data[frame_indices[i]])))
            # else:
            #     return video
                video.append(Image.open(io.BytesIO(video_data[frame_indices[i]])))


        return video

def ImageToHdf5(clip,hdf5_path):

    # import pdb
    #
    # pdb.set_trace()
    dst_dir_path = Path(hdf5_path[:-5])
    dst_dir_path.mkdir(exist_ok=True)

    for n,img in enumerate(clip):

        img=Image.fromarray(img)
        img.save(os.path.join(dst_dir_path,'image_{:3d}.jpg'.format(n)),'JPEG',quality=80)

    success_code=1
    try:
        with h5py.File(hdf5_path, 'w') as f:
            dtype = h5py.special_dtype(vlen='uint8')
            video = f.create_dataset('video',
                                     (len(clip),),
                                     dtype=dtype)

        for i, file_path in enumerate(sorted(dst_dir_path.glob('*.jpg'))):
            with file_path.open('rb') as f:
                data = f.read()
            with h5py.File(hdf5_path, 'r+') as f:
                video = f['video']
                video[i] = np.frombuffer(data, dtype='uint8')

        for file_path in dst_dir_path.glob('*.jpg'):
            file_path.unlink()
        dst_dir_path.rmdir()
    except:
        success_code=0
        pass
    return success_code