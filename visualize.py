"""Visualization script for PATNet Cross-Domain Few-Shot Segmentation"""
import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model.patnet import PATNetwork
from data.dataset import FSSDataset


def visualize(model, dataloader, nshot, num_samples=5, save_dir='./visualizations'):
    """Visualize segmentation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            # Get batch data
            query_img = batch['query_img'].cuda()
            query_mask = batch['query_mask'].cuda()
            support_imgs = batch['support_imgs'].cuda()
            support_masks = batch['support_masks'].cuda()
            query_name = batch['query_name'][0]
            
            # Forward pass
            pred_mask = model.predict_mask_nshot(query_img, support_imgs, support_masks, nshot)
            
            # Convert to numpy for visualization
            query_img_np = query_img[0].cpu().permute(1, 2, 0).numpy()
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            query_img_np = std * query_img_np + mean
            query_img_np = np.clip(query_img_np, 0, 1)
            
            query_mask_np = query_mask[0].cpu().numpy()
            pred_mask_np = pred_mask[0].cpu().numpy()
            
            # Support image
            support_img_np = support_imgs[0, 0].cpu().permute(1, 2, 0).numpy()
            support_img_np = std * support_img_np + mean
            support_img_np = np.clip(support_img_np, 0, 1)
            
            support_mask_np = support_masks[0, 0].cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Support
            axes[0, 0].imshow(support_img_np)
            axes[0, 0].set_title('Support Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(support_mask_np, cmap='gray')
            axes[0, 1].set_title('Support Mask', fontsize=12)
            axes[0, 1].axis('off')
            
            # Overlay support mask on image
            support_overlay = support_img_np.copy()
            support_overlay[support_mask_np > 0.5] = [1, 0, 0]  # Red overlay
            axes[0, 2].imshow(support_overlay)
            axes[0, 2].set_title('Support Overlay', fontsize=12)
            axes[0, 2].axis('off')
            
            # Row 2: Query
            axes[1, 0].imshow(query_img_np)
            axes[1, 0].set_title('Query Image', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(query_mask_np, cmap='gray')
            axes[1, 1].set_title('Ground Truth', fontsize=12)
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(pred_mask_np, cmap='gray')
            axes[1, 2].set_title('Prediction', fontsize=12)
            axes[1, 2].axis('off')
            
            # Calculate IoU for this sample
            intersection = np.logical_and(query_mask_np > 0.5, pred_mask_np > 0.5).sum()
            union = np.logical_or(query_mask_np > 0.5, pred_mask_np > 0.5).sum()
            iou = intersection / (union + 1e-6) * 100
            
            plt.suptitle(f'Sample {idx+1} - IoU: {iou:.2f}%', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, f'sample_{idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved visualization: {save_path} (IoU: {iou:.2f}%)')
    
    print(f'\nAll visualizations saved to: {save_dir}')


def visualize_comparison(model, dataloader, nshot, num_samples=5, save_dir='./visualizations'):
    """Create comparison visualization with overlay"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            query_img = batch['query_img'].cuda()
            query_mask = batch['query_mask'].cuda()
            support_imgs = batch['support_imgs'].cuda()
            support_masks = batch['support_masks'].cuda()
            
            pred_mask = model.predict_mask_nshot(query_img, support_imgs, support_masks, nshot)
            
            # Convert to numpy
            query_img_np = query_img[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            query_img_np = std * query_img_np + mean
            query_img_np = np.clip(query_img_np, 0, 1)
            
            query_mask_np = query_mask[0].cpu().numpy()
            pred_mask_np = pred_mask[0].cpu().numpy()
            
            # Create overlay comparison
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(query_img_np)
            axes[0].set_title('Query Image', fontsize=14)
            axes[0].axis('off')
            
            # Ground truth overlay (green)
            gt_overlay = query_img_np.copy()
            gt_overlay[query_mask_np > 0.5] = gt_overlay[query_mask_np > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
            axes[1].imshow(gt_overlay)
            axes[1].set_title('Ground Truth (Green)', fontsize=14)
            axes[1].axis('off')
            
            # Prediction overlay (blue)
            pred_overlay = query_img_np.copy()
            pred_overlay[pred_mask_np > 0.5] = pred_overlay[pred_mask_np > 0.5] * 0.5 + np.array([0, 0, 1]) * 0.5
            axes[2].imshow(pred_overlay)
            axes[2].set_title('Prediction (Blue)', fontsize=14)
            axes[2].axis('off')
            
            # Combined: GT=Green, Pred=Blue, Overlap=Cyan
            combined = query_img_np.copy()
            gt_only = np.logical_and(query_mask_np > 0.5, pred_mask_np <= 0.5)
            pred_only = np.logical_and(query_mask_np <= 0.5, pred_mask_np > 0.5)
            overlap = np.logical_and(query_mask_np > 0.5, pred_mask_np > 0.5)
            
            combined[gt_only] = combined[gt_only] * 0.5 + np.array([1, 0, 0]) * 0.5  # Red: missed
            combined[pred_only] = combined[pred_only] * 0.5 + np.array([1, 1, 0]) * 0.5  # Yellow: false positive
            combined[overlap] = combined[overlap] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green: correct
            axes[3].imshow(combined)
            axes[3].set_title('Comparison\n(Green=Correct, Red=Missed, Yellow=FP)', fontsize=12)
            axes[3].axis('off')
            
            # Calculate IoU
            intersection = np.logical_and(query_mask_np > 0.5, pred_mask_np > 0.5).sum()
            union = np.logical_or(query_mask_np > 0.5, pred_mask_np > 0.5).sum()
            iou = intersection / (union + 1e-6) * 100
            
            plt.suptitle(f'Sample {idx+1} - IoU: {iou:.2f}%', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'comparison_{idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved: {save_path} (IoU: {iou:.2f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PATNet Results')
    parser.add_argument('--datapath', type=str, default='./CDFSL')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['fss', 'deepglobe'])
    parser.add_argument('--load', type=str, required=True, help='Path to trained model')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations')
    parser.add_argument('--mode', type=str, default='both', choices=['basic', 'comparison', 'both'])
    args = parser.parse_args()
    
    # Create save directory with benchmark name
    save_dir = os.path.join(args.save_dir, args.benchmark)
    
    # Initialize dataset
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, 1, 0, 'test', args.nshot)
    
    # Build model
    model = PATNetwork(args.backbone)
    model.eval()
    model.cuda()
    
    # Load weights
    print(f'Loading model from: {args.load}')
    model.load_state_dict(torch.load(args.load))
    
    print(f'\nVisualizing {args.num_samples} samples from {args.benchmark} dataset...\n')
    
    if args.mode in ['basic', 'both']:
        visualize(model, dataloader, args.nshot, args.num_samples, save_dir)
    
    if args.mode in ['comparison', 'both']:
        # Reinitialize dataloader for comparison
        dataloader = FSSDataset.build_dataloader(args.benchmark, 1, 0, 'test', args.nshot)
        visualize_comparison(model, dataloader, args.nshot, args.num_samples, save_dir)
    
    print(f'\n✅ Visualization complete! Check: {save_dir}')
