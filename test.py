r""" Few-Shot Semantic Segmentation testing code — PASCAL VOC 1-way 5-shot """
import argparse

import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test PATNet """
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATNet Testing — 1-way 5-shot PASCAL VOC')
    parser.add_argument('--datapath', type=str, default='./dataset',
                        help='Root dataset path (contains VOCdevkit/)')
    parser.add_argument('--benchmark', type=str, default='pascal',
                        choices=['pascal', 'fss', 'deepglobe', 'isic', 'lung', 'chick'])
    parser.add_argument('--logpath', type=str, default='./')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='',
                        help='Path to trained model checkpoint (best_model.pt)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    model = PATNetwork(args.backbone)
    model.eval()
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))

    Evaluator.initialize()

    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    Logger.info('Test benchmark: %s (%d batches, %d-shot)' % (args.benchmark, len(dataloader_test), args.nshot))

    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)

    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
