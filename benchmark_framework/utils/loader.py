import io

import h5py
from PIL import Image
# import mxnet as mx

class ImageLoaderMx(object):
    def __call__(self, path):
        return mx.image.imdecode(open(str(path),'rb').read()).asnumpy()

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
        try:
            with h5py.File(video_path, 'r') as f:
                video_data = f['video']

                video = []
                for i in range(len(frame_indices)):
                # if i < len(video_data):
                #     video.append(Image.open(io.BytesIO(video_data[frame_indices[i]])))
                # else:
                #     return video
                    video.append(Image.open(io.BytesIO(video_data[frame_indices[i]-1])))
        except:
            print(video_path)
            print(frame_indices)
            print(len(video_data))
            raise

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        try:
            with h5py.File(video_path, 'r') as f:

                flow_data = []
                for flow in self.flows:
                    flow_data.append(f['video_{}'.format(flow)])

                video = []
                for i in range(len(frame_indices)):
                    if frame_indices[i] < len(flow_data[0]):
                        frame = [
                            Image.open(io.BytesIO(video_data[frame_indices[i]]))
                            for video_data in flow_data
                        ]
                        frame.append(frame[-1])  # add dummy data into third channel
                        video.append(Image.merge('RGB', frame))
        except:
            print(video_path)
            pass
        
        return video