r"""Generate high-resolution paper figures for PATNet few-shot segmentation.

Outputs per query sample (saved to ./gambarforpaper/sample_XX_STEM/):

  Individual images at ORIGINAL resolution:
    a_query_image.png               - Query image
    b_ground_truth_mask.png         - Ground truth binary mask
    c_patnet_1shot_prediction.png   - PATNet 1-shot prediction
    d_patnet_5shot_prediction.png   - PATNet 5-shot prediction
    support_image_1..N.png          - Support images
    support_mask_1..N.png           - Support masks

  Combined figure:
    combined_figure.png             - All panels in one figure (300 DPI)

Usage:
  python generate_paper_figures.py
  python generate_paper_figures.py --datapath ./dataset --load ./logs/patnet_pascal.log/best_model.pt
  python generate_paper_figures.py --img_size 512 --benchmark chick
"""
import os
import argparse
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model.patnet import PATNetwork
from data.dataset import FSSDataset
from common import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_ids(img_dir, mask_dir):
    """Return sorted image stems that have a matching mask file."""
    imgs = glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))
    ids = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join(mask_dir, stem + ".png")):
            ids.append(stem)
    return sorted(ids)


def get_support_stems(all_ids, support_ids, query_stem):
    """Replicate DatasetChick support-selection logic."""
    sup = [s for s in support_ids if s != query_stem]
    if len(sup) < len(support_ids):
        candidates = support_ids + [s for s in all_ids if s not in support_ids]
        for c in candidates:
            if c != query_stem and c not in sup:
                sup.append(c)
                break
    return sup


def load_original_image(img_dir, stem):
    """Load full-resolution RGB image."""
    for ext in (".jpg", ".png"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"No image found for stem '{stem}' in {img_dir}")


def load_original_mask(mask_dir, stem):
    """Load full-resolution binary mask as uint8 array {0,1}."""
    p = os.path.join(mask_dir, stem + ".png")
    m = np.array(Image.open(p).convert("L"))
    return (m > 0).astype(np.uint8)


def mask_to_image(mask, target_size=None):
    """Binary mask -> uint8 image (0/255).  target_size = (W, H) for resize."""
    img = (mask * 255).astype(np.uint8)
    if target_size is not None:
        img = np.array(Image.fromarray(img).resize(target_size, Image.NEAREST))
    return img


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def create_combined_figure(query_img, gt_mask, pred_1shot, pred_5shot,
                           support_imgs, support_masks, outpath,
                           sample_name=""):
    """Compose a single high-res figure with all panels.

    Layout (rows x cols):
      Row 0  : (a) Query | (b) GT | (c) 1-shot | (d) 5-shot
      Row 1  : Support Image 1 .. N
      Row 2  : Support Mask  1 .. N
    """
    n_sup = len(support_imgs)
    n_cols = max(4, n_sup)
    n_rows = 3

    q_arr = np.array(query_img)
    q_h, q_w = q_arr.shape[:2]
    cell_w = 6                          # inches per cell
    cell_h = cell_w * q_h / q_w        # keep aspect ratio
    fig_w = n_cols * cell_w + 1.5
    fig_h = n_rows * cell_h + 2.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.18, wspace=0.06)

    # Row 0 – main results
    labels = ['(a) Query Image', '(b) Ground Truth',
              '(c) PATNet 1-shot', '(d) PATNet 5-shot']
    row0_imgs = [q_arr, gt_mask, pred_1shot, pred_5shot]
    row0_cmap = [None, 'gray', 'gray', 'gray']

    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        if row0_cmap[j]:
            ax.imshow(row0_imgs[j], cmap=row0_cmap[j], vmin=0, vmax=255)
        else:
            ax.imshow(row0_imgs[j])
        ax.set_title(labels[j], fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    # Row 1 – support images
    for j in range(n_sup):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(np.array(support_imgs[j]))
        ax.set_title(f'Support Image {j + 1}', fontsize=11, pad=8)
        ax.axis('off')

    # Row 2 – support masks
    for j in range(n_sup):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(support_masks[j], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Support Mask {j + 1}', fontsize=11, pad=8)
        ax.axis('off')

    if sample_name:
        fig.suptitle(sample_name, fontsize=16, fontweight='bold', y=0.99)

    plt.savefig(outpath, dpi=300, bbox_inches='tight',
                pad_inches=0.3, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate high-res paper figures for PATNet')
    parser.add_argument('--datapath', type=str, default='./dataset')
    parser.add_argument('--load', type=str, default='./logs/patnet_pascal.log/best_model.pt')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50'])
    parser.add_argument('--img_size', type=int, default=512,
                        help='Model input size (predictions at this res)')
    parser.add_argument('--outdir', type=str, default='./gambarforpaper')
    parser.add_argument('--benchmark', type=str, default='chick')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    utils.fix_randseed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---- Load model -------------------------------------------------------
    model = PATNetwork(args.backbone)
    model.eval()
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(
        torch.load(args.load, map_location=device, weights_only=True))
    print(f'Model loaded from {args.load}')

    # ---- Original-resolution images (for saving) --------------------------
    root = os.path.join(args.datapath, args.benchmark)
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "segmentations")
    all_ids = collect_ids(img_dir, msk_dir)
    support_ids_5 = all_ids[:5]

    print(f'Found {len(all_ids)} images')
    print(f'5-shot supports: {support_ids_5}')
    print(f'Output: {os.path.abspath(args.outdir)}')

    # ---- Dataloader (5-shot provides all supports) ------------------------
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
    dataloader = FSSDataset.build_dataloader(
        args.benchmark, 1, 0, fold=0, split='test', shot=5)

    # ---- Inference + save -------------------------------------------------
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = utils.to_cuda(batch)

            # Predictions (model resolution)
            pred_1 = model.module.predict_mask_nshot(batch, nshot=1)
            pred_5 = model.module.predict_mask_nshot(batch, nshot=5)
            pred_1_np = pred_1[0].detach().cpu().numpy()
            pred_5_np = pred_5[0].detach().cpu().numpy()

            # Original-resolution query
            q_stem = all_ids[idx]
            q_img_orig = load_original_image(img_dir, q_stem)
            q_mask_orig = load_original_mask(msk_dir, q_stem)
            orig_w, orig_h = q_img_orig.size

            # Support stems (replicates dataset logic)
            sup_stems = get_support_stems(all_ids, support_ids_5, q_stem)

            # Output directory for this sample
            sample_dir = os.path.join(args.outdir,
                                      f'sample_{idx:02d}_{q_stem}')
            os.makedirs(sample_dir, exist_ok=True)

            # -- (a) Query image – original resolution --
            q_img_orig.save(
                os.path.join(sample_dir, 'a_query_image.png'))

            # -- (b) Ground truth mask – original resolution --
            gt_img = mask_to_image(q_mask_orig)
            Image.fromarray(gt_img).save(
                os.path.join(sample_dir, 'b_ground_truth_mask.png'))

            # -- (c) PATNet 1-shot prediction – resized to original res --
            pred_1_img = mask_to_image(pred_1_np, (orig_w, orig_h))
            Image.fromarray(pred_1_img).save(
                os.path.join(sample_dir, 'c_patnet_1shot_prediction.png'))

            # -- (d) PATNet 5-shot prediction – resized to original res --
            pred_5_img = mask_to_image(pred_5_np, (orig_w, orig_h))
            Image.fromarray(pred_5_img).save(
                os.path.join(sample_dir, 'd_patnet_5shot_prediction.png'))

            # -- Support images & masks – original resolution --
            sup_imgs_orig = []
            sup_masks_img = []
            for s_i, s_stem in enumerate(sup_stems):
                s_img = load_original_image(img_dir, s_stem)
                s_mask = load_original_mask(msk_dir, s_stem)

                s_img.save(
                    os.path.join(sample_dir,
                                 f'support_image_{s_i + 1}.png'))
                s_mask_img = mask_to_image(s_mask)
                Image.fromarray(s_mask_img).save(
                    os.path.join(sample_dir,
                                 f'support_mask_{s_i + 1}.png'))

                sup_imgs_orig.append(s_img)
                sup_masks_img.append(s_mask_img)

            # -- Combined figure (300 DPI) --
            create_combined_figure(
                q_img_orig, gt_img, pred_1_img, pred_5_img,
                sup_imgs_orig, sup_masks_img,
                os.path.join(sample_dir, 'combined_figure.png'),
                sample_name=q_stem,
            )

            print(f'[{idx + 1}/{len(all_ids)}] Saved → {sample_dir}')

    print(f'\nDone! All figures saved to: {os.path.abspath(args.outdir)}')


if __name__ == '__main__':
    main()
