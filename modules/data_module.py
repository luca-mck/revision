# define data module

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from utils.io_utils import get_files, get_annotation, get_image_size, get_label_map

# define data module
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class ObjectDetectionDataSet(Dataset):
    def __init__(self, data_tuples, tranformations=None, mapping=None) -> None:
        super().__init__()
        self.data_tuples = data_tuples
        self.transformations = tranformations
        self.mapping = mapping

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        img = read_image(self.data_tuples[index][0])
        annotations = get_annotation(self.data_tuples[index][1], self.mapping)
        if self.transformations:
            img = self.transformations(img / 255.0)
        return img, annotations


class ObjectDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        image_folder,
        annotation_folder,
        transformations=None,
        test_suffix=None,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=16,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = self.get_transformations(transformations)
        self.train_val_split = 0.5
        self.test_suffix = test_suffix

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        
        if test_suffix:
            self.image_folder_test = image_folder + test_suffix
            self.annotation_folder_test = annotation_folder + test_suffix

    @staticmethod
    def get_transformations(transformation_list):
        transformations = []
        for transformation in transformation_list:
            if transformation == "normalize":
                transformations.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                )
        return transforms.Compose(transformations)

    def prepare_data(self) -> None:
        self.data_tuples = get_files(self.image_folder, self.annotation_folder)[:4]
        self.mapping = get_label_map([files[1] for files in self.data_tuples])
        self.image_size, self.resizing = get_image_size([files[0] for files in self.data_tuples])
        if self.resizing:
            raise ValueError("Images have different dimensions")
        train_len = int(len(self.data_tuples) * self.train_val_split)
        val_len = len(self.data_tuples) - train_len
        self.seq_lengths = [train_len, val_len]
        if self.test_suffix:
            self.data_tuples_test = get_files(
                self.image_folder_test, self.annotation_folder_test
            )
        else:
            self.data_tuples_test = []

    def setup(self, stage: str = "fit"):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset = ObjectDetectionDataSet(
                self.data_tuples, self.transform, mapping=self.mapping
            )

            self.train_set, self.val_set = random_split(
                dataset, self.seq_lengths, torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = ObjectDetectionDataSet(
                self.data_tuples_test, mapping=self.mapping
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            #num_workers=8,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            shuffle=True,
            #num_workers=8,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    #def test_dataloader(self):
    #    return DataLoader(
    #        self.test_set,
    #        batch_size=self.test_batch_size,
    #        shuffle=True,
    #        #num_workers=8,
    #        collate_fn=lambda batch: tuple(zip(*batch)),
    #    )
