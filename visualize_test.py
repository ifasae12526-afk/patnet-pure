r"""Visualize a single test episode for PATNet.

Saves an output image showing: support images, query image, ground-truth mask,
predicted mask, and overlay comparisons.

Usage example:
python visualize_test.py --datapath ./dataset --load ./logs/experiment_3.log/best_model.pt --nshot 1 --idx 0 --outdir ./vis_out

"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from model.patnet import PATNetwork
from data.dataset import FSSDataset
from common import utils

# image normalization parameters (match FSSDataset.initialize)
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(img_t):
    """Convert a torch tensor image (C,H,W) normalized to a PIL Image (RGB)."""
    if isinstance(img_t, torch.Tensor):
        img = img_t.detach().cpu().numpy()
    else:
        img = np.array(img_t)
    # img shape C,H,W
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    # unnormalize
    img = (img * IMG_STD[None, None, :]) + IMG_MEAN[None, None, :]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def mask_to_pil(mask_t, color=(255, 0, 0), alpha=128):
    """Convert binary mask tensor (H,W) to RGBA PIL image with color overlay."""
    if isinstance(mask_t, torch.Tensor):
        mask = mask_t.detach().cpu().numpy()
    else:
        mask = np.array(mask_t)
    mask = (mask > 0).astype(np.uint8) * 255
    mask_img = Image.new('RGBA', (mask.shape[1], mask.shape[0]), (0, 0, 0, 0))
    overlay = Image.new('RGBA', (mask.shape[1], mask.shape[0]), color + (alpha,))
    mask_pil = Image.fromarray(mask).convert('L')
    mask_img.paste(overlay, (0, 0), mask_pil)
    return mask_img


def overlay_images(base_pil, mask_pil):
    out = base_pil.convert('RGBA')
    out.paste(mask_pil, (0, 0), mask_pil)
    return out


def visualize_episode(batch, pred_mask, outpath, nshot=1):
    """Compose and save visualization for a single episode batch.

    batch: dictionary with keys 'query_img', 'query_mask', 'support_imgs', 'support_masks'
    pred_mask: tensor [H,W] predicted binary mask
    outpath: output file path
    """
    query_img = batch['query_img']  # [C,H,W]
    query_mask = batch['query_mask']  # [H,W]
    support_imgs = batch['support_imgs']  # [shot, C, H, W]
    support_masks = batch['support_masks']  # [shot, H, W]

    # Convert to PIL
    q_pil = tensor_to_pil(query_img)
    q_gt_mask = mask_to_pil(query_mask, color=(0, 255, 0), alpha=120)
    q_pred_mask = mask_to_pil(pred_mask, color=(255, 0, 0), alpha=120)
    q_gt_overlay = overlay_images(q_pil, q_gt_mask)
    q_pred_overlay = overlay_images(q_pil, q_pred_mask)

    # Support images
    support_pils = []
    support_mask_pils = []
    for s_idx in range(nshot):
        s_img = support_imgs[s_idx]
        s_mask = support_masks[s_idx]
        support_pils.append(tensor_to_pil(s_img))
        support_mask_pils.append(overlay_images(tensor_to_pil(s_img), mask_to_pil(s_mask, color=(0, 0, 255), alpha=120)))

    # Build canvas: supports (2 rows: image + masked), then query, gt overlay, pred overlay
    # Determine widths/heights
    w, h = q_pil.size
    pad = 8
    cols = max(nshot, 1) + 3  # supports + query + gt + pred
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = 2 * h + 3 * pad

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

    x = pad
    # Place support images (top row) and their overlays (bottom row)
    for i in range(nshot):
        canvas.paste(support_pils[i].resize((w, h)), (x, pad))
        canvas.paste(support_mask_pils[i].resize((w, h)), (x, pad + h + pad))
        x += w + pad

    # Place placeholders if fewer supports than columns
    # Place query original
    canvas.paste(q_pil.resize((w, h)), (x, pad)); x += w + pad
    # Place GT overlay
    canvas.paste(q_gt_overlay.resize((w, h)), (x, pad)); x += w + pad
    # Place Pred overlay
    canvas.paste(q_pred_overlay.resize((w, h)), (x, pad)); x += w + pad

    # Save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    canvas.save(outpath)


def main():
    parser = argparse.ArgumentParser(description='Visualize PATNet test episode')
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--idx', type=int, default=0, help='Episode index to visualize')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='./vis_out')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    args = parser.parse_args()

    # Setup
    utils.fix_randseed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = PATNetwork(args.backbone)
    model.eval()
    model = nn.DataParallel(model)
    model.to(device)

    if not os.path.exists(args.load):
        raise FileNotFoundError(f'Model checkpoint not found: {args.load}')
    model.load_state_dict(torch.load(args.load, map_location=device))

    # Dataset
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader('custom', args.bsz, 0, fold=0, split='test', shot=args.nshot)

    # Fetch episode
    # dataloader yields batches; find requested index
    for i, batch in enumerate(dataloader):
        if i != args.idx:
            continue
        # Move tensors to device
        # Note: keep CPU copies for visualization; use .to(device) for model inputs
        batch_gpu = {}
        batch_cpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
                batch_cpu[k] = v.detach().cpu()
            else:
                batch_gpu[k] = v
                batch_cpu[k] = v

        with torch.no_grad():
            pred_mask = model.module.predict_mask_nshot(batch_gpu, nshot=args.nshot)
        # pred_mask: [B,H,W] ; we use first item
        pred = pred_mask[0].detach().cpu()

        # For visualization we need sample tensors in expected shapes
        # Convert query_img support_imgs etc to CPU tensors (C,H,W)
        vis_batch = {
            'query_img': batch_cpu['query_img'][0],
            'query_mask': batch_cpu['query_mask'][0],
            'support_imgs': batch_cpu['support_imgs'][0],
            'support_masks': batch_cpu['support_masks'][0]
        }

        outpath = os.path.join(args.outdir, f'episode_{args.idx:04d}.png')
        visualize_episode(vis_batch, pred, outpath, nshot=args.nshot)
        print('Saved visualization to', outpath)
        break


if __name__ == '__main__':
    main()
