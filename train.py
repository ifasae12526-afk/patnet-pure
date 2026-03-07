r""" PATNet training (validation) code — pure PASCAL VOC setup.

Training setup:
  - Train: pascal (VOC2012, fold=4, all 20 classes)
  - Validation: pascal (VOC2012, fold-based validation)

Algorithm: PATNet (ResNet50 backbone, 4D correlation, CrossEntropyLoss).
"""
import sys
sys.path.insert(0, "../")

import argparse
import time

import torch.optim as optim
import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def train(epoch, model, dataloader, optimizer, training, label=''):
    r""" Train PATNet """

    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        logit_mask = model(
            batch['query_img'],
            batch['support_imgs'].squeeze(1),
            batch['support_masks'].squeeze(1)
        )
        pred_mask = logit_mask.argmax(dim=1)

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    tag = label if label else ('Training' if training else 'Validation')
    average_meter.write_result(tag, epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATNet Training — Pure PASCAL VOC')

    # Data paths
    parser.add_argument('--datapath', type=str, default='./dataset',
                        help='Root dataset path (contains VOCdevkit/)')

    # Training benchmarks
    parser.add_argument('--benchmark_train', type=str, default='pascal',
                        choices=['pascal', 'fss', 'deepglobe', 'isic', 'lung'])
    parser.add_argument('--benchmark_val', type=str, default='pascal',
                        choices=['pascal', 'fss', 'deepglobe', 'isic', 'lung'])

    parser.add_argument('--logpath', type=str, default='patnet_pascal')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--bsz_val', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--niter', type=int, default=8)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--val_fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--img_size', type=int, default=512)

    # For backward compatibility with Logger
    parser.add_argument('--benchmark', type=str, default='pascal')

    args = parser.parse_args()
    args.benchmark = args.benchmark_train  # Logger uses args.benchmark

    Logger.initialize(args, training=True)

    model = PATNetwork(args.backbone)
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    # ── Train dataset ──
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark_train, args.bsz,
                                                  args.nworker, args.fold, 'trn')
    Logger.info('Train: %s (%d batches)' % (args.benchmark_train, len(dataloader_trn)))

    # ── Validation dataset ──
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark_val, args.bsz_val,
                                                  args.nworker, args.val_fold, 'val')
    Logger.info('Validation: %s (%d batches)' % (args.benchmark_val, len(dataloader_val)))

    best_val_miou = float('-inf')
    start_time = time.time()

    for epoch in range(args.niter):

        # Phase 1: Training
        trn_loss, trn_miou, trn_fb_iou = train(
            epoch, model, dataloader_trn, optimizer, training=True,
            label='Train-%s' % args.benchmark_train)

        # Phase 2: Validation
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(
                epoch, model, dataloader_val, optimizer, training=False,
                label='Val-%s' % args.benchmark_val)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        # Tensorboard (same keys as SAM2CDFSS for easy comparison)
        Logger.tbd_writer.add_scalars('loss', {'trn': trn_loss, 'val': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('miou', {'trn': trn_miou, 'val': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('fb_iou', {'trn': trn_fb_iou, 'val': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        elapsed = time.time() - start_time
        if (epoch + 1) % 5 == 0:
            Logger.info('[time] elapsed %.2fh | best val mIoU %.2f' % (elapsed / 3600, best_val_miou))

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
