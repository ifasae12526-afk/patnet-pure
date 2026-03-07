r""" Example test script untuk custom dataset dengan XML annotation """
import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from data.custom_utils import validate_dataset_structure


def test_custom_dataset(model, dataloader, nshot):
    r""" Test PATNet dengan custom dataset """
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test Custom Dataset', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PATNet dengan Custom Dataset')
    parser.add_argument('--datapath', type=str, default='./data/custom_dataset', help='Path ke custom dataset root')
    parser.add_argument('--logpath', type=str, default='./')
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--nworker', type=int, default=0, help='Number of workers')
    parser.add_argument('--load', type=str, default='', help='Path to trained model checkpoint')
    parser.add_argument('--nshot', type=int, default=1, help='Number of support images')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    parser.add_argument('--img_size', type=int, default=400, help='Input image size')
    args = parser.parse_args()

    args.benchmark = 'custom'
    if args.load == '':
        args.load = 'no_model'

    Logger.initialize(args, training=False)
    Logger.info(f'Testing Custom Dataset: {args.datapath}')
    Logger.info(f'N-shot: {args.nshot}')

    model = PATNetwork(args.backbone)
    model.eval()
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info(f'# available GPUs: {torch.cuda.device_count()}')
    model = nn.DataParallel(model)
    model.to(device)

    if args.load == '':
        Logger.warning('No pretrained model specified! Using random initialization.')
    else:
        if os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load))
            Logger.info(f'Loaded model from: {args.load}')
        else:
            Logger.error(f'Model file not found: {args.load}')
            sys.exit(1)

    Evaluator.initialize()
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)

    Logger.info(f'Validating dataset structure at: {args.datapath}')
    dataset_info = validate_dataset_structure(args.datapath)
    if not dataset_info['valid']:
        Logger.error('Invalid dataset structure!')
        for issue in dataset_info['issues']:
            Logger.error(f'  - {issue}')
        sys.exit(1)

    Logger.info('Dataset validation passed:')
    Logger.info(f'  Classes: {len(dataset_info["classes"])}')
    Logger.info(f'  Total Images: {dataset_info["num_images"]}')
    Logger.info(f'  Total Annotations: {dataset_info["num_annotations"]}')

    try:
        dataloader_test = FSSDataset.build_dataloader(
            'custom', args.bsz, args.nworker, fold=0, split='test', shot=args.nshot
        )
        Logger.info('Custom dataset loaded successfully')
        Logger.info(f'Dataset size: {len(dataloader_test.dataset)} episodes')
    except Exception as e:
        Logger.error(f'Failed to load custom dataset: {str(e)}')
        sys.exit(1)

    Logger.info('Starting evaluation...')
    with torch.no_grad():
        test_miou, test_fb_iou = test_custom_dataset(model, dataloader_test, args.nshot)

    Logger.info(f'mIoU: {test_miou.item():5.2f} \t FB-IoU: {test_fb_iou.item():5.2f}')
    Logger.info('==================== Finished Testing ====================')
