r""" Custom dataset with XML annotation for few-shot semantic segmentation.

Supports Roboflow Pascal VOC XML that contains <polygon> with x1,y1,x2,y2,... points.

Directory structure expected:
dataset/
    ├── images/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class2/
    └── annotations/
        ├── class1/
        │   ├── img1.xml
        │   └── ...
        └── class2/

XML example (Roboflow VOC-like):
<annotation>
  <size><width>...</width><height>...</height></size>
  <object>
    <name>class_name</name>
    <bndbox>...</bndbox>          (optional / fallback)
    <polygon>
      <x1>...</x1><y1>...</y1>
      <x2>...</x2><y2>...</y2>
      ...
    </polygon>
  </object>
</annotation>
"""
import os
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from PIL import ImageDraw


class DatasetCustom(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=None):
        self.split = split
        self.benchmark = 'custom'
        self.shot = shot
        self.num = num

        self.datapath = datapath
        self.img_dir = os.path.join(datapath, 'images')
        self.anno_dir = os.path.join(datapath, 'annotations')

        # categories = subfolders under images/
        self.categories = sorted([
            d for d in os.listdir(self.img_dir)
            if os.path.isdir(os.path.join(self.img_dir, d))
        ])

        self.class_ids = list(range(len(self.categories)))
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        total_samples = sum(len(imgs) for imgs in self.img_metadata_classwise.values())
        self.num = num if num is not None else total_samples

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(
            query_name, support_names, class_sample
        )

        # apply image transform
        query_img_t = self.transform(query_img)

        # resize masks to transformed image size (nearest)
        query_mask_t = torch.as_tensor(query_mask, dtype=torch.long)  # (H,W)
        query_mask_t = F.interpolate(
            query_mask_t.unsqueeze(0).unsqueeze(0).float(),
            query_img_t.size()[-2:],
            mode='nearest'
        ).squeeze(0).squeeze(0).long()

        support_imgs_t = torch.stack([self.transform(simg) for simg in support_imgs])

        support_masks_t = []
        for smask in support_masks:
            smask_t = torch.as_tensor(smask, dtype=torch.long)
            smask_t = F.interpolate(
                smask_t.unsqueeze(0).unsqueeze(0).float(),
                support_imgs_t.size()[-2:],
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
            support_masks_t.append(smask_t)
        support_masks_t = torch.stack(support_masks_t)  # (shot,H,W)

        batch = {
            'query_img': query_img_t,
            'query_mask': query_mask_t,
            'query_name': query_name,
            'support_imgs': support_imgs_t,
            'support_masks': support_masks_t,
            'support_names': support_names,
            'class_id': torch.tensor(class_sample)
        }
        return batch

    def build_img_metadata_classwise(self):
        img_metadata_classwise = defaultdict(list)

        for class_idx, category in enumerate(self.categories):
            cat_img_dir = os.path.join(self.img_dir, category)
            if not os.path.exists(cat_img_dir):
                print(f"Warning: Directory {cat_img_dir} tidak ditemukan")
                continue

            img_paths = sorted([
                path for path in glob.glob(os.path.join(cat_img_dir, '*'))
                if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
            ])

            img_metadata_classwise[class_idx] = img_paths

        return img_metadata_classwise

    def sample_episode(self, idx):
        # random class
        class_sample = np.random.choice(list(self.img_metadata_classwise.keys()))

        query_imgs = self.img_metadata_classwise[class_sample]
        if len(query_imgs) < 1:
            raise ValueError(
                f"Class {self.categories[class_sample]} tidak memiliki images. "
                f"Pastikan direktori images/{self.categories[class_sample]}/ tidak kosong."
            )

        query_idx = np.random.choice(range(len(query_imgs)))
        query_name = query_imgs[query_idx]

        available_indices = [i for i in range(len(query_imgs)) if i != query_idx]
        if len(available_indices) == 0:
            raise ValueError(
                f"Class {self.categories[class_sample]} hanya memiliki 1 image. "
                f"Butuh minimal {self.shot + 1} images untuk nshot={self.shot} (1 query + {self.shot} support)."
            )

        if len(available_indices) < self.shot:
            import warnings
            warnings.warn(
                f"Class {self.categories[class_sample]} hanya punya {len(available_indices)} kandidat support. "
                f"Sampling dengan replacement: support bisa duplicate. "
                f"Disarankan tambah lebih banyak images per class.",
                UserWarning
            )
            selected_indices = np.random.choice(available_indices, self.shot, replace=True)
        else:
            selected_indices = np.random.choice(available_indices, self.shot, replace=False)

        support_names = [query_imgs[i] for i in selected_indices]
        return query_name, support_names, class_sample

    def load_frame(self, query_name, support_names, class_id):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_mask = self.load_mask_from_xml(query_name, class_id)
        support_masks = [self.load_mask_from_xml(name, class_id) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    @staticmethod
    def _parse_polygon_points(poly_elem):
        """
        Roboflow VOC polygon format:
          <polygon><x1>...</x1><y1>...</y1><x2>...</x2><y2>...</y2>...</polygon>

        Returns: list[(x,y)] sorted by index
        """
        xs = {}
        ys = {}
        for e in list(poly_elem):
            tag = e.tag.lower().strip()
            if tag.startswith('x'):
                try:
                    idx = int(tag[1:])
                    xs[idx] = float(e.text)
                except Exception:
                    continue
            elif tag.startswith('y'):
                try:
                    idx = int(tag[1:])
                    ys[idx] = float(e.text)
                except Exception:
                    continue

        keys = sorted(set(xs.keys()) & set(ys.keys()))
        pts = [(xs[k], ys[k]) for k in keys]
        return pts

    def load_mask_from_xml(self, img_path, class_id):
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        class_name = self.categories[class_id]

        xml_path = os.path.join(self.anno_dir, class_name, img_base + '.xml')
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"XML annotation file not found: {xml_path}\n"
                f"Make sure annotation exists for {class_name}/{img_name}"
            )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise Exception(f"Error parsing XML {xml_path}: {e}")

        # image size: prefer XML size, fallback to reading image
        size_node = root.find('size')
        if size_node is not None and size_node.find('width') is not None and size_node.find('height') is not None:
            img_width = int(float(size_node.find('width').text))
            img_height = int(float(size_node.find('height').text))
        else:
            im = Image.open(img_path).convert('RGB')
            img_width, img_height = im.size

        # build blank mask
        mask_img = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask_img)

        # Fill polygons for objects of this class
        found_any_polygon = False
        found_any_bbox = False

        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is None:
                continue
            if (name_node.text or '').strip() != class_name:
                continue

            # 1) Prefer polygon
            poly = obj.find('polygon')
            if poly is not None:
                pts = self._parse_polygon_points(poly)
                if len(pts) >= 3:
                    # clamp points
                    pts_clamped = []
                    for x, y in pts:
                        x = max(0, min(int(round(x)), img_width - 1))
                        y = max(0, min(int(round(y)), img_height - 1))
                        pts_clamped.append((x, y))
                    draw.polygon(pts_clamped, outline=1, fill=1)
                    found_any_polygon = True
                    continue  # done for this object

            # 2) Fallback: bbox (ONLY if no polygon)
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    xmin = max(0, min(xmin, img_width - 1))
                    ymin = max(0, min(ymin, img_height - 1))
                    xmax = max(0, min(xmax, img_width - 1))
                    ymax = max(0, min(ymax, img_height - 1))

                    # draw rectangle as mask region
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=1, fill=1)
                    found_any_bbox = True
                except Exception:
                    pass

        mask = np.array(mask_img, dtype=np.uint8)  # 0/1

        # If polygon exists in file but none matched class, mask will be empty.
        # That's expected for class-specific episodes.
        return mask

    def read_mask(self, img_path, class_id):
        """Alias untuk kompatibilitas"""
        return self.load_mask_from_xml(img_path, class_id)
