# data/lombok.py
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import cv2

class DatasetLandslide(Dataset):
    def __init__(self, root, fold=0, transform=None, split='trn', shot=1):
        self.root = root
        self.split = split
        self.shot = shot
        self.transform = transform
        self.benchmark = 'landslide'
        self.class_ids = [0]

        split_file = os.path.join(root, "splits", f"{split}.txt")
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.img_dir = os.path.join(root, "img")
        self.msk_dir = os.path.join(root, "label")

    def _read_tif_rgb(self, path):
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = np.transpose(arr, (1,2,0))  # HWC
        return arr

    def _read_mask(self, path):
        with rasterio.open(path) as src:
            m = src.read(1)  # (H,W)
        m = (m > 0).astype(np.uint8)  # pastikan 0/1
        return m

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1) pilih query
        q_id = random.choice(self.ids)
        q_img = self._read_tif_rgb(os.path.join(self.img_dir, f"{q_id}.tif"))
        q_msk = self._read_mask(os.path.join(self.msk_dir, f"{q_id}.tif"))

        # 2) pilih support (beda file dari query)
        candidates = [x for x in self.ids if x != q_id]
        support_ids = random.sample(candidates, k=self.shot)

        s_imgs, s_msks = [], []
        for s_id in support_ids:
            s_img = self._read_tif_rgb(os.path.join(self.img_dir, f"{s_id}.tif"))
            s_msk = self._read_mask(os.path.join(self.msk_dir, f"{s_id}.tif"))
            s_imgs.append(s_img)
            s_msks.append(s_msk)

        # 3) resize + toTensor + normalize pakai transform bawaan repo
        # transform repo di FSSDataset adalah torchvision transform untuk PIL,
        # jadi paling gampang: konversi dulu ke uint8 PIL atau pakai cv2 + manual tensor.
        # (saya buat manual agar stabil utk tif)

        def img_to_tensor(img_hwc):
            img = cv2.resize(img_hwc, (400, 400), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            mx = img.max()
            if mx > 0:
                img = img / mx
            img = torch.from_numpy(img).permute(2,0,1)  # CHW
            # normalisasi ImageNet (sesuai FSSDataset) :contentReference[oaicite:3]{index=3}
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            return (img - mean) / std

        def mask_to_tensor(msk_hw):
            m = cv2.resize(msk_hw, (400, 400), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
            return torch.from_numpy(m).long()  # H,W

        q_img_t = img_to_tensor(q_img)
        q_msk_t = mask_to_tensor(q_msk)

        s_imgs_t = torch.stack([img_to_tensor(x) for x in s_imgs], dim=0)        # [shot,3,H,W]
        s_msks_t = torch.stack([mask_to_tensor(x) for x in s_msks], dim=0)       # [shot,H,W]

        batch = {
            "query_img": q_img_t,
            "query_mask": q_msk_t,
            "support_imgs": s_imgs_t,
            "support_masks": s_msks_t,
            "class_id": torch.tensor(0).long(),
        }
        return batch
