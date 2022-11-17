import argparse
import numpy as np
import os
import shutil
import pandas as pd

import pytorch_lightning as pl
import torch
import torchvision.transforms as T

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy


# Download 'https://www.kaggle.com/competitions/histopathologic-cancer-detection/'
cancer_csv_filename = "./histopathologic-cancer-detection/train_labels.csv"
dataset_folder = "./histopathologic-cancer-detection/"
np.random.seed(42)


def random_image_list():
    train_imgs_orig = os.listdir(dataset_folder + "train")
    selected_image_list = []
    for img in np.random.choice(train_imgs_orig, 10000):
        selected_image_list.append(img)
    return selected_image_list


def preprocessing(image_list):
    selected_image_list = image_list
    np.random.shuffle(selected_image_list)
    cancer_train_idx = selected_image_list[:8000]
    cancer_test_idx = selected_image_list[8000:]

    if os.path.exists(dataset_folder + "train_dataset"):
        shutil.rmtree(dataset_folder + "train_dataset")

    if os.path.exists(dataset_folder + "test_dataset"):
        shutil.rmtree(dataset_folder + "test_dataset")

    os.mkdir(dataset_folder + "train_dataset")
    for fname in cancer_train_idx:
        src = os.path.join(dataset_folder + "train", fname)
        dst = os.path.join(dataset_folder + "train_dataset", fname)
        shutil.copyfile(src, dst)

    os.mkdir(dataset_folder + "test_dataset")
    for fname in cancer_test_idx:
        src = os.path.join(dataset_folder + "train", fname)
        dst = os.path.join(dataset_folder + "test_dataset", fname)
        shutil.copyfile(src, dst)


def data_loader(image_list):
    cancer_labels = pd.read_csv(cancer_csv_filename)

    data_T_train = T.Compose(
        [
            T.CenterCrop(32),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
        ]
    )

    data_T_test = T.Compose(
        [
            T.CenterCrop(32),
            T.ToTensor(),
        ]
    )

    selected_image_labels = pd.DataFrame()
    id_list = []
    label_list = []

    for img in image_list:
        label_tuple = cancer_labels.loc[cancer_labels["id"] == img.split(".")[0]]
        id_list.append(label_tuple["id"].values[0])
        label_list.append(label_tuple["label"].values[0])

    selected_image_labels["id"] = id_list
    selected_image_labels["label"] = label_list

    img_label_dict = {
        k: v for k, v in zip(selected_image_labels.id, selected_image_labels.label)
    }

    train_set = LoadCancerDataset(
        data_folder=dataset_folder + "train_dataset",
        transform=data_T_train,
        dict_labels=img_label_dict,
    )

    test_set = LoadCancerDataset(
        data_folder=dataset_folder + "test_dataset",
        transform=data_T_test,
        dict_labels=img_label_dict,
    )

    return train_set, test_set


def valid_model(model, test_dataloader):
    model.eval()
    preds = []
    for batch_i, (data, target) in enumerate(test_dataloader):
        data, target = data.cuda(), target.cuda()
        output = model.cuda()(data)
        pr = output[:, 1].detach().cpu().numpy()
        for i in pr:
            preds.append(i)
    return preds


class LoadCancerDataset(Dataset):
    def __init__(
        self,
        data_folder,
        transform=T.Compose([T.CenterCrop(32), T.ToTensor()]),
        dict_labels={},
    ):
        self.data_folder = data_folder
        self.list_image_files = [s for s in os.listdir(data_folder)]
        self.transform = transform
        self.dict_labels = dict_labels
        self.labels = [dict_labels[i.split(".")[0]] for i in self.list_image_files]

    def __len__(self):
        return len(self.list_image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.list_image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.list_image_files[idx].split(".")[0]

        label = self.dict_labels[img_name_short]
        return image, label


class CNNImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.fully_connected_1 = nn.Linear(in_features=16 * 16 * 6, out_features=1000)
        self.fully_connected_2 = nn.Linear(in_features=1000, out_features=500)
        self.fully_connected_3 = nn.Linear(in_features=500, out_features=250)
        self.fully_connected_4 = nn.Linear(in_features=250, out_features=120)
        self.fully_connected_5 = nn.Linear(in_features=120, out_features=60)
        self.fully_connected_6 = nn.Linear(in_features=60, out_features=2)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input):
        output = self.conv_layer1(input)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv_layer2(output)
        output = self.relu2(output)
        output = output.view(-1, 6 * 16 * 16)
        output = self.fully_connected_1(output)
        output = self.fully_connected_2(output)
        output = self.fully_connected_3(output)
        output = self.fully_connected_4(output)
        output = self.fully_connected_5(output)
        output = self.fully_connected_6(output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log("train_accuracy", train_accuracy, prog_bar=True)
        self.log("train_loss", loss)
        return {"loss": loss, "train_accuracy": train_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log("test_accuracy", test_accuracy)
        return {"test_loss": loss, "test_accuracy": test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def binary_accuracy(self, outputs, targets):
        _, outputs = torch.max(outputs, 1)
        correct_results_sum = (outputs == targets).sum().float()
        acc = correct_results_sum / targets.shape[0]
        return acc

    def predict_step(self, batch, batch_idx):
        return self(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN for PyTorch')
    parser.add_argument('--test', action='store_true', default=False, help='Train or Test')

    args = parser.parse_args()
    if not args.test:
    
        print("1.Starting Preparation of Data")
        selected_image_list = random_image_list()
        preprocessing(selected_image_list)
        train_set, test_set = data_loader(selected_image_list)

        print("2.Starting Data Loader")
        batch_size = 256
        train_dataloader = DataLoader(
            train_set, batch_size, num_workers=2, pin_memory=True, shuffle=True
        )

        print("3.Starting Learning")
        model = CNNImageClassifier()
        trainer = pl.Trainer(accelerator="gpu", devices=1)
        # trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
        trainer.fit(model, train_dataloaders=train_dataloader)

        print("4.Starting Model Evaluation")
        test_dataloader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)
        trainer.test(dataloaders=test_dataloader)
        preds = valid_model(model=model, test_dataloader=test_dataloader)

        test_preds = pd.DataFrame(
            {"imgs": test_set.list_image_files, "labels": test_set.labels, "preds": preds}
        )
        test_preds.to_csv("test_preds.csv", index=False)
    else:
        test_preds = pd.read_csv("test_preds.csv")
        test_preds["imgs"] = test_preds["imgs"].apply(lambda x: x.split(".")[0])
        test_preds.head()
        test_preds["predictions"] = 1
        test_preds.loc[test_preds["preds"] < 0, "predictions"] = 0
        test_preds.shape
        test_preds.head()        
        score = len(
            np.where(test_preds["labels"] == test_preds["predictions"])[0]
        ) / test_preds.shape[0]
        print(f"Score is {score}")
