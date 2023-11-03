import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoverTargetTransform

import numpy as np

from torch.utils.data import Dataset
import cv2
import pydicom
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

ROOT_PATH = './mimic-cxr-jpg'

def get_datamodule(args, mode):
    if mode == "pretrain":
        return PretrainMIMICDataModule(args)
    elif mode == "discover":
        return DiscoverMIMICDataModule(args)


all_train = os.path.join("./ncdcsvs/sub1/05", "all_train.csv")
labeled_train = os.path.join("./ncdcsvs/sub1/05", "labeled_train.csv")
labeled_test = os.path.join("./ncdcsvs/sub1/05", "labeled_test.csv")
unlabeled_train = os.path.join("./ncdcsvs/sub1/05", "unlabeled_train.csv")
unlabeled_test = os.path.join("./ncdcsvs/sub1/05", "unlabeled_test.csv")


def get_path(x):
    return os.path.join(ROOT_PATH,"p"+str(int(x["subject_id"]))[0:2], "p"+str(int(x["subject_id"])), "s"+str(int(x["study_id"])), str(x["dicom_id"])+".jpg")



class MIMICDataset(Dataset):
    def __init__(self, csv, transform):
        self.csv = csv
        self.images = []
        self.texts = []
        self.labels = []
        self.preprocess = transform
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        df = pd.read_csv(self.csv)
        df['path'] = df.apply(get_path, axis=1)

        labels_index = ['Atelectasis','Edema',  'Lung Opacity', 'No Finding',
                        'Consolidation', 'Pneumonia', 'Pneumothorax']
                        # 'Cardiomegaly','Enlarged Cardiomediastinum', 'Fracture','Pleural Effusion']
        for index, row in tqdm(df.iterrows()):
            text = str(row['attrs'])
            # self.texts.append(text)
            self.texts.append(self.tokenizer(text,
                                        padding='max_length',
                                        max_length=128,
                                        truncation=True,
                                        return_tensors="pt"))
            img_path = row['path']
            x = cv2.imread(str(img_path), 0)
            x = Image.fromarray(x).convert('RGB')

            img = self.preprocess(x)
            self.images.append(img)

            # labeled classes
            if row['Atelectasis'] == 1:
                label = torch.tensor(labels_index.index('Atelectasis'))
            elif row['Lung Lesion'] == 1:
                label = torch.tensor(labels_index.index('Lung Lesion'))
            elif row['Lung Opacity'] == 1:
                label = torch.tensor(labels_index.index('Lung Opacity'))
            elif row['No Finding'] == 1:
                label = torch.tensor(labels_index.index('No Finding'))
            elif row['Consolidation'] == 1:     #unlabeled class 1
                label = torch.tensor(labels_index.index('Consolidation'))
            elif row['Edema'] == 1:
                label = torch.tensor(labels_index.index('Edema'))
            elif row['Pneumonia'] == 1:
                label = torch.tensor(labels_index.index('Pneumonia'))
            elif row['Pneumothorax'] == 1:
                label = torch.tensor(labels_index.index('Pneumothorax'))
            elif row['Cardiomegaly'] == 1:      #unlabeled class 2
                label = torch.tensor(labels_index.index('Cardiomegaly'))
            elif row['Enlarged Cardiomediastinum'] == 1:
                label = torch.tensor(labels_index.index('Enlarged Cardiomediastinum'))
            elif row['Fracture'] == 1:
                label = torch.tensor(labels_index.index('Fracture'))
            elif row['Pleural Effusion'] == 1:
                label = torch.tensor(labels_index.index('Pleural Effusion'))

            self.labels.append(label)


    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        return len(self.images)


class PretrainMIMICDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.transform_train = get_transforms("unsupervised", args.dataset)
        self.transform_val = get_transforms("eval", args.dataset)
        self.train_dataset = MIMICDataset(csv=labeled_train, transform=self.transform_train)
        self.val_dataset = MIMICDataset(csv=labeled_test, transform=self.transform_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )



class DiscoverMIMICDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)

        self.train_dataset = MIMICDataset(csv=all_train, transform=self.transform_train)
        val_subset_unlab_train = MIMICDataset(csv=unlabeled_train, transform=self.transform_val)
        val_subset_unlab_test = MIMICDataset(csv=unlabeled_test, transform=self.transform_val)
        val_subset_lab_test = MIMICDataset(csv=labeled_test, transform=self.transform_val)
        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]


    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in self.val_datasets
        ]



class DiscoverDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)


