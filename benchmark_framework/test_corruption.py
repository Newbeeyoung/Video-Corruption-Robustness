import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

from models import build_model
from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
from utils.temporal_transform import *
from utils.video_dataset import VideoDataset
from utils.dataset_config import get_dataset_config
from utils.loader import VideoLoaderHDF5
from opts import arg_parser

from pdb import set_trace

def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print("Number of GPUs:", torch.cuda.device_count())

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)

    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes = num_classes
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    model = model.cuda()
    model.eval()
    model = torch.nn.DataParallel(model)

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> creating model '{}'".format(arch_name))

    model.cuda()

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    # Data loading code
    # data_list = os.path.join(args.datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}".format(args.num_clips))

    temporal_transform = []
    # temporal_transform.append(Uniform_Sampling(n_samples=args.groups))
    temporal_transform.append(Uniform_Sampling_Fixoffset(n_samples=args.groups))
    temporal_transform = Compose(temporal_transform)

    video_path_formatter = (lambda root_path, label, video_id: root_path /
                                                               label / '{}.hdf5'.format(video_id))


    corruption_dict = {
                        "shot_noise": [1, 2, 3, 4, 5],
                        'motion_blur': [1,2,3,4,5],
                        'rain': [1, 2, 3, 4, 5],
                        'contrast': [1, 2, 3, 4, 5],
                        'brightness': [1, 2, 3, 4, 5],
                        'saturate': [1, 2, 3, 4, 5],
                        'fog':[1,2,3,4,5],
                        'bit_error_hdf5': [100000,50000, 30000, 20000, 10000],
                        'packet_loss_hdf5':[1,2,3,4,6],
                        'h265_crf_hdf5':[27,33,39,45,51],
                        'h264_crf_hdf5':[23,30,37,44,51],
                        'frame_rate': [6, 9, 12, 16, 20]
        }

    corruption_transform = None

    Non_crop_list=['shot_noise','rain','contrast','brightness','saturate']
    crop_list=['motion_blur','packet_loss_hdf5','bit_error_hdf5','h265_crf_hdf5','h265_abr_hdf5','frame_rate']
    for key, severity_levels in corruption_dict.items():
        args.corruption = key
        for severity in severity_levels:
            print("Corruption:", key, "Severity:", severity)

            val_video_path = Path("{}/kinetics/mini_kinetics200-c/{}/{}".format(args.datadir,key,severity))

            val_annotation_path=Path("data/mini_kinetics/mini-kinetics200_{}_{}.json".format(key,severity))
            print(val_annotation_path)
            augments = []
            if key in crop_list:
                augments += [
                    GroupScale(scale_size),
                    GroupCenterCrop(args.input_size)
                ]

            augments += [
                Stack(threed_data=args.threed_data),
                ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
                GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
            ]

            augmentor = transforms.Compose(augments)

            val_dataset = VideoDataset(val_video_path,
                                          val_annotation_path,
                                          'validation',
                                          corruption_transform=corruption_transform,
                                          spatial_transform=augmentor,
                                          temporal_transform=temporal_transform,
                                          target_transform=None,
                                          video_loader=VideoLoaderHDF5(),
                                          video_path_formatter=video_path_formatter)

            data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                         workers=args.workers)

            log_folder = os.path.join(args.logdir, arch_name)
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            batch_time = AverageMeter()
            if args.evaluate:
                logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
                top1 = AverageMeter()
                top5 = AverageMeter()
            else:
                logfile = open(os.path.join(log_folder, 'test_{}crops_{}clips_{}.csv'.format(
                    args.num_crops, args.num_clips, args.input_size)), 'w')

            total_outputs = 0
            outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
            # switch to evaluate mode
            model.eval()
            total_batches = len(data_loader)
            with torch.no_grad(), tqdm(total=total_batches) as t_bar:
                end = time.time()
                for i, (video, label) in enumerate(data_loader):
                    output = eval_a_batch(video, model, num_clips=args.num_clips, num_crops=args.num_crops,
                                          threed_data=args.threed_data)
                    if args.evaluate:
                        label = label.cuda(non_blocking=True)
                        # measure accuracy
                        prec1, prec5 = accuracy(output, label, topk=(1, 5))
                        top1.update(prec1[0], video.size(0))
                        top5.update(prec5[0], video.size(0))
                        output = output.data.cpu().numpy().copy()
                        batch_size = output.shape[0]
                        outputs[total_outputs:total_outputs + batch_size, :] = output
                    else:
                        # testing, store output to prepare csv file
                        output = output.data.cpu().numpy().copy()
                        batch_size = output.shape[0]
                        outputs[total_outputs:total_outputs + batch_size, :] = output
                        predictions = np.argsort(output, axis=1)
                        for ii in range(len(predictions)):
                            # preds = [id_to_label[str(pred)] for pred in predictions[ii][::-1][:5]]
                            temp = predictions[ii][::-1][:5]
                            preds = [str(pred) for pred in temp]
                            print("{};{}".format(label[ii], ";".join(preds)), file=logfile)
                    total_outputs += video.shape[0]
                    batch_time.update(time.time() - end)
                    end = time.time()
                    t_bar.update(1)

                outputs = outputs[:total_outputs]
                print("Predict {} videos.".format(total_outputs), flush=True)
                np.save(os.path.join(log_folder, '{}_{}_{}_details.npy'.format(
                    "val" if args.evaluate else "test", key,severity)), outputs)

            if args.evaluate:
                print(
                    'Val@{}({}) (# corruption ={}, #severity ={}, # crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
                        args.input_size, scale_size, args.corruption, severity, args.num_crops, args.num_clips,
                        top1.avg, top5.avg),
                    flush=True)
                print(
                    'Val@{}({}) (# corruption ={}, #severity ={}, # crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
                        args.input_size, scale_size, args.corruption, severity, args.num_crops, args.num_clips,
                        top1.avg, top5.avg),
                    flush=True, file=logfile)

            logfile.close()



if __name__ == '__main__':
    main()
