r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.fss import DatasetFSS
from data.deepglobe import DatasetDeepglobe
from data.isic import DatasetISIC
from data.lung import DatasetLung
from data.chick import DatasetChick
try:
    from data.custom import DatasetCustom
except ImportError:
    DatasetCustom = None
try:
    from data.landslide import DatasetLandslide
except ImportError:
    DatasetLandslide = None


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, episodes_per_epoch=200):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,
            'chick': DatasetChick,
        }
        if DatasetCustom is not None:
            cls.datasets['custom'] = DatasetCustom
        if DatasetLandslide is not None:
            cls.datasets['landslide'] = DatasetLandslide

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.episodes_per_epoch = episodes_per_epoch

        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(cls.img_mean, cls.img_std)
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        # Pass episodes_per_epoch for datasets that support it (e.g., chick)
        try:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform,
                                               split=split, shot=shot,
                                               episodes_per_epoch=cls.episodes_per_epoch)
        except TypeError:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform,
                                               split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
