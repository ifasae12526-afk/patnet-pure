r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetLung(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'Lung Segmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def _mask_to_stem(self, mask_path):
        """Convert mask filename to image stem robustly.
        
        Removes '_mask' suffix if present, then returns basename without extension.
        E.g., '/path/to/MCUCXR_0_mask.png' -> 'MCUCXR_0'
        """
        basename = os.path.basename(mask_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Remove '_mask' suffix if present
        if name_without_ext.endswith('_mask'):
            stem = name_without_ext[:-5]  # Remove '_mask'
        else:
            stem = name_without_ext
        
        return stem

    def _find_image_by_stem(self, stem):
        """Find image file in img_path by stem, trying common extensions and glob fallback.
        
        Args:
            stem: image filename stem (without extension)
            
        Returns:
            Full path to image file, or None if not found
        """
        # Try common extensions first
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = os.path.join(self.img_path, stem + ext)
            if os.path.exists(candidate):
                return candidate
        
        # Fallback: glob for stem.*
        glob_pattern = os.path.join(self.img_path, stem + '.*')
        matches = glob.glob(glob_pattern)
        if matches:
            return matches[0]  # Return first match
        
        return None

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        # Robustly find query image by stem
        query_stem = self._mask_to_stem(query_name)
        query_img_path = self._find_image_by_stem(query_stem)
        if query_img_path is None:
            raise FileNotFoundError(f"No image found for mask: {query_name} (stem: {query_stem})")
        query_img = Image.open(query_img_path).convert('RGB')

        # Robustly find support images by stem
        support_imgs = []
        for support_name in support_names:
            support_stem = self._mask_to_stem(support_name)
            support_img_path = self._find_image_by_stem(support_stem)
            if support_img_path is None:
                raise FileNotFoundError(f"No image found for mask: {support_name} (stem: {support_stem})")
            support_imgs.append(Image.open(support_img_path).convert('RGB'))

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    # Check if corresponding image exists
                    mask_stem = self._mask_to_stem(img_path)
                    if self._find_image_by_stem(mask_stem) is not None:
                        img_metadata_classwise[cat] += [img_path]
                    else:
                        print(f"[WARNING] Skipping mask (no matching image): {img_path}")
        
        return img_metadata_classwise
