import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)

        return out


class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)
        self.loop2=LoopPadding(n_samples)

    def __call__(self, frame_indices):

        if len(frame_indices)<self.n_samples:
        #     # frame_indices=self.loop2(frame_indices)
            frame_indices=key_frame_sampling(len(frame_indices),self.n_samples)

        n_frames = len(frame_indices)

        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))
            if len(sample) < self.size:
                # out.append(self.loop(sample))
                out.append(key_frame_sampling(len(frame_indices),self.size))
                # break
            else:
                out.append(sample)

        return out

def key_frame_sampling(key_cnt,frame_size):
    factor=frame_size*1.0/key_cnt
    index=[int(j/factor)+1 for j in range(frame_size)]
    return index

class Uniform_Sampling(object):

    def __init__(self, size=1, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(n_samples)

    def __call__(self, frame_indices):

        if len(frame_indices)<=self.n_samples:
            out=key_frame_sampling(len(frame_indices),self.n_samples)

        else:
            n_frames = len(frame_indices)
            stride = max(
                1, math.floor((n_frames -  self.size) / self.n_samples))

            out = []
            for n,begin_index in enumerate(frame_indices[::stride]):
                if len(out) >= self.n_samples:
                    break

                end_index=begin_index+stride-1
                if n==self.n_samples-1:
                    end_index=n_frames-1

                sample=random.randint(begin_index,end_index)
                out.append(sample)

            # if len(out)<self.n_samples:
            #     out=self.loop(out)
        print(out)
        return out

class Uniform_Sampling_Fixoffset(object):
    def __init__(self, size=1, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(n_samples)

    def __call__(self, frame_indices):

        if len(frame_indices)<=self.n_samples:
            out=key_frame_sampling(len(frame_indices),self.n_samples)

        else:
            n_frames = len(frame_indices)
            stride = max(
                1, math.floor((n_frames -  self.size) / self.n_samples))

            offset=random.randint(0,stride-1)

            out = []
            for n,begin_index in enumerate(frame_indices[::stride]):
                if len(out) >= self.n_samples:
                    break

                sample=begin_index+offset
                out.append(sample)

            # if len(out)<self.n_samples:
            #     out=self.loop(out)
        # print(out)
        return out

class Uniform_Sampling_Multiclips(object):

    def __init__(self, size=1, n_clips=8,n_samples=1,average=True):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(n_samples)
        self.average=average
        self.n_clips=n_clips

    def __call__(self, frame_indices):
        # if len(frame_indices)<self.n_samples:
        #     frame_indices=self.loop(frame_indices)

        if len(frame_indices)<self.n_clips:
            frame_indices=key_frame_sampling(len(frame_indices),self.n_clips)

        n_frames = len(frame_indices)
        stride = max(
            1, math.floor((n_frames-1 - self.size) / self.n_samples))

        if self.average:
            # n_clips=n_frames-frame_indices[::stride][-1]+1
            n_clips=self.n_clips
        else:
            n_clips=1
        outs=[]
        for begin in range(n_clips):
            out = []
            for begin_index in frame_indices[begin::stride]:
                if len(out) >= self.n_samples:
                    break

                sample=begin_index
                out.append(sample)

            if len(out)<self.n_samples:
                # out=self.loop(out)
                out=key_frame_sampling(max(out),self.n_samples)
            outs.append(out)

        # print(outs)
        return outs

class RandomStack(object):
    def __init__(self,size):
        self.size=size

    def __call__(self, frame_indices):
        sample=random.randint(0,frame_indices[-1]+1)
        out=[]
        for i in range(self.size):
            out.append(sample)

        return out

class SlidingWindow(object):

    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


class TemporalSubsampling(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]


class Shuffle(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i:(i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices