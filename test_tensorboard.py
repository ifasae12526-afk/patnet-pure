"""Test script with TensorboardX visualization for PATNet"""
import argparse
import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import torchvision.utils as vutils

from model.patnet import PATNetwork
from common.logger import AverageMeter
from common.evaluation import Evaluator
from data.dataset import FSSDataset


def test_with_tensorboard(model, dataloader, nshot, writer, benchmark):
    """Test and log results to TensorBoard"""
    model.eval()
    
    average_meter = AverageMeter(dataloader.dataset)
    
    # Mean and std for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            query_img = batch['query_img']
            query_mask = batch['query_mask']
            support_imgs = batch['support_imgs']
            support_masks = batch['support_masks']
            class_id = batch['class_id']
            
            # Predict
            pred_mask = model.predict_mask_nshot(batch, nshot)
            
            # Evaluate
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
            average_meter.update(area_inter, area_union, class_id, loss=None)
            
            # Log images for first 20 samples
            if idx < 20:
                # Denormalize images
                query_img_denorm = query_img[0].cpu() * std + mean
                query_img_denorm = torch.clamp(query_img_denorm, 0, 1)
                
                support_img_denorm = support_imgs[0, 0].cpu() * std + mean
                support_img_denorm = torch.clamp(support_img_denorm, 0, 1)
                
                # Get masks
                gt_mask = query_mask[0].cpu().float()
                pred = pred_mask[0].cpu().float()
                support_mask_viz = support_masks[0, 0].cpu().float()
                
                # Calculate IoU for this sample
                inter = (gt_mask > 0.5) & (pred > 0.5)
                union = (gt_mask > 0.5) | (pred > 0.5)
                iou = inter.sum().float() / (union.sum().float() + 1e-6) * 100
                
                # Create colored overlay
                # GT = Green channel, Pred = Blue channel
                gt_colored = torch.zeros(3, gt_mask.shape[0], gt_mask.shape[1])
                gt_colored[1] = gt_mask  # Green for GT
                
                pred_colored = torch.zeros(3, pred.shape[0], pred.shape[1])
                pred_colored[2] = pred  # Blue for prediction
                
                # Overlay on image
                alpha = 0.4
                query_with_gt = query_img_denorm * (1 - alpha * gt_mask) + gt_colored * alpha
                query_with_pred = query_img_denorm * (1 - alpha * pred) + pred_colored * alpha
                
                # Support with mask overlay
                support_colored = torch.zeros(3, support_mask_viz.shape[0], support_mask_viz.shape[1])
                support_colored[0] = support_mask_viz  # Red for support mask
                support_with_mask = support_img_denorm * (1 - alpha * support_mask_viz) + support_colored * alpha
                
                # Log images
                writer.add_image(f'{benchmark}/sample_{idx:03d}/1_query_image', query_img_denorm, 0)
                writer.add_image(f'{benchmark}/sample_{idx:03d}/2_support_image', support_img_denorm, 0)
                writer.add_image(f'{benchmark}/sample_{idx:03d}/3_support_mask', support_with_mask, 0)
                writer.add_image(f'{benchmark}/sample_{idx:03d}/4_ground_truth', query_with_gt, 0)
                writer.add_image(f'{benchmark}/sample_{idx:03d}/5_prediction', query_with_pred, 0)
                
                # Log per-sample IoU
                writer.add_scalar(f'{benchmark}/per_sample_iou', iou.item(), idx)
            
            # Progress
            avg_iou, fb_iou = average_meter.compute_iou()
            print(f'[Batch: {idx+1:04d}/{len(dataloader):04d}] mIoU: {avg_iou:.2f}  |  FB-IoU: {fb_iou:.2f}')
    
    # Final metrics
    avg_iou, fb_iou = average_meter.compute_iou()
    
    # Log final metrics
    writer.add_scalar(f'{benchmark}/final_mIoU', avg_iou, 0)
    writer.add_scalar(f'{benchmark}/final_FB-IoU', fb_iou, 0)
    
    return avg_iou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PATNet with TensorboardX visualization')
    parser.add_argument('--datapath', type=str, default='./CDFSL')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['fss', 'deepglobe', 'isic', 'lung'])
    parser.add_argument('--logpath', type=str, default='./logs_test')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()

    # Create log directory
    logdir = os.path.join(args.logpath, f'{args.benchmark}_test')
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(logdir)
    
    print('\n:=========== Test with TensorboardX ===========')
    print(f'|             datapath: {args.datapath}')
    print(f'|            benchmark: {args.benchmark}')
    print(f'|              logpath: {logdir}')
    print(f'|                  bsz: {args.bsz}')
    print(f'|                 load: {args.load}')
    print(f'|                nshot: {args.nshot}')
    print(f'|             backbone: {args.backbone}')
    print(':================================================\n')

    # Initialize dataset - batch size must be 1 for proper evaluation
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, 1, args.nworker, args.fold, 'test', args.nshot)

    # Build model
    model = PATNetwork(args.backbone)
    model.eval()
    model.cuda()

    # Load weights (handle DataParallel saved models)
    print(f'Loading model from: {args.load}')
    state_dict = torch.load(args.load, weights_only=False)
    # Remove 'module.' prefix if model was saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # Test with TensorBoard logging
    miou, fb_iou = test_with_tensorboard(model, dataloader, args.nshot, writer, args.benchmark)
    
    writer.close()
    
    print(f'\n*** Final Results ***')
    print(f'mIoU: {miou:.2f}      FB-IoU: {fb_iou:.2f}')
    print(f'\n✅ TensorBoard logs saved to: {logdir}')
    print(f'Run: tensorboard --logdir="{logdir}" --port=6006')
