import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchmetrics
import pandas as pd
import cv2
from PIL import Image
from torchvision.models import resnet18
import pdb
import argparse




learning_rate = 0.0001

train_loss = []
val_loss = []
train_acc = []
val_acc = []

class AudioDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.data_paths[index])  # 画像を読み込む
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 画像をRGB形式に変換
        image = Image.fromarray(image)  # NumPy ndarrayからPIL画像に変換
        if self.transform is not None:
            image = self.transform(image)  # 画像に変換関数を適用

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

class CNNModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)  # 例としてResNet-18を使用
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize='true')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc(pred,y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        train_loss.append(loss.item())
        train_acc.append(self.train_acc(pred, y).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('val/acc', self.val_acc(pred,y), prog_bar=True, logger=True)

        val_loss.append(loss.item())
        val_acc.append(self.val_acc(pred, y).item())
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('test/acc', self.test_acc(pred, y), prog_bar=True, logger=True)
        return {'pred':torch.argmax(pred, dim=-1), 'target':y}

    def test_epoch_end(self, outputs) -> None:
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp['pred'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(10), columns=range(10))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='gray_r').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

def add_path_label():
    data = []

    path_img = '../dataset/mfcc_img'
    path_label = '../training.csv'


    for img_path in os.listdir(f'{path_img}'):
        data.append([f'{path_img}/{img_path}', img_path, 0])

    df = pd.DataFrame(data, columns=['path', 'filename', 'label'])

    df.head()

    training = pd.read_csv(os.path.join(path_label))
    label_ = training['label'].values
    name = training['path'].values

    for i, path in enumerate(name):
        path = path.replace('dataset/train/', '')
        path = path.replace('dataset/test/', '')
        path = path.replace('.wav', '')
        for k, filename_ in enumerate(df['filename']):
            if path in filename_:
                try:
                    df.loc[k, 'label'] = label_[i]
                except KeyError as e:
                    pdb.set_trace()

    #print(df.head())
    return df


def add_test_path_label(path_to_truth):
    data = []

    path_img = '../dataset/mfcc_test_img/'
    path_label = path_to_truth


    for img_path in os.listdir(f'{path_img}'):
        data.append([f'{path_img}/{img_path}', img_path, 0])

    df = pd.DataFrame(data, columns=['path', 'filename', 'label'])

    df.head()

    training = pd.read_csv(os.path.join(path_label))
    label_ = training['label'].values
    name = training['path'].values

    for i, path in enumerate(name):
        path = path.replace('dataset/train/', '')
        path = path.replace('dataset/test/', '')
        path = path.replace('.wav', '')
        for k, filename_ in enumerate(df['filename']):
            if path in filename_:
                try:
                    df.loc[k, 'label'] = label_[i]
                except KeyError as e:
                    pdb.set_trace()

    #print(df.head())
    return df


def los_acc_plot():
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    t_tl = np.linspace(0,len(train_loss), len(train_loss))
    t_vl = np.linspace(0,len(val_loss), len(val_loss))
    t_ta = np.linspace(0,len(train_acc), len(train_acc))
    t_va = np.linspace(0,len(val_acc), len(val_acc))

    # print(t_tl)
    # print("-------------------")
    # print(train_loss)
    # print("-------------------")
    # print(len(train_loss))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid()
    ax1.plot(t_tl, train_loss, color="red", label="train_loss")
    ax1.legend(loc = 0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid()
    ax2.plot(t_ta, train_acc, color="blue", label="train_acc")
    ax2.legend(loc = 0)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid()
    ax3.plot(t_vl, val_loss, color="green", label="val_loss")
    ax3.legend(loc = 0)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.grid()
    ax4.plot(t_va, val_acc, color="orange", label="val_acc")
    ax4.legend(loc = 0)
    fig.tight_layout()
    plt.savefig('loss and accuracy(1)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help='path of test data')
    args = parser.parse_args()

    training = add_path_label()
    data_paths = training["path"].values # 音声データファイルのパス
    labels = training['label'].values # ラベル
    num_classes = len(set(labels))
    #import pdb; pdb.set_trace()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #print(data_paths)
    #print(labels)

    dataset = AudioDataset(data_paths, labels, transform)

    # Train/Validation 分割
    val_size = int(len(dataset)*0.2)
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            torch.Generator().manual_seed(20200616))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)


    model = CNNModel(num_classes)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model=model, dataloaders=val_loader)


    # モデルを評価（検証データに対する評価）
    result = trainer.validate(model, dataloaders=val_loader)
    print(result)

    if args.path_to_truth:
        # Test Dataset の作成
        test = add_path_label(args.path_to_truth)
        test_data_paths = test["path"].values
        test_labels = test['label'].values
        test_dataset = AudioDataset(test_data_paths, test_labels, transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        trainer.test(model=model, dataloaders=test_loader)

    """val_accuracy = trainer.test(dataloaders=val_loader)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")"""
    #import pdb; pdb.set_trace()
    los_acc_plot()

if __name__ == '__main__':
    main()

