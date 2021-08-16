#%%
import setuptools
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import seaborn as sns; sns.set()
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision as tv
from sklearn.metrics import r2_score, mean_squared_error
from adabelief_pytorch import AdaBelief
import torch.nn.functional as F
import os
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import gc
from torchsummary import summary

seed = 999

# seeds and flags
np.random.seed(seed)
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.determinstic = True

sns.set_style("dark")

#%%
# Dataset preparation
SLICE = 2000
MAPS = np.load('channels_uncond_10k.npy').T
MAPS = MAPS[:SLICE, :]
normed_maps = MAPS.reshape(-1, 1, 64, 64) / np.abs(MAPS).max()
normed_maps = torch.from_numpy(normed_maps).float()
dummy = torch.randn_like(normed_maps)

c, dim = 1, 64

dataset = torch.utils.data.TensorDataset(normed_maps, dummy)

batch_size = 100
lr = 1e-3
# C, H, W = 1, 28, 28
C, H, W = 1, 64, 64
in_size = (C, normed_maps.shape[-2], normed_maps.shape[-1])
epochs = 1000

# %%
# data module
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=64, split=None, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.split = split

    def prepare_data(self):
        # download only
        self.dataset = self.dataset

    def setup(self, stage=None):
        # train/valid/test split
        # and assign to use in dataloaders via self
        train_set, valid_set, test_set = torch.utils.data.random_split(self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed))

        if stage == 'fit' or stage is None:
            self.train_set = train_set
            self.valid_set = valid_set

        if stage == 'test' or stage is None:
            self.test_set = test_set

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

#%%
class AE(pl.LightningModule):
    def __init__(self, in_dims):
        super().__init__()
        self.in_size = in_dims
        self.l_r = 0.02
        self.channels = 8

        # encoder model
        self.encoder = nn.Sequential(
            # conv_1
            nn.Conv2d(in_channels=self.in_size[0], out_channels=self.channels*16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.SELU(),
            # conv_2
            nn.Conv2d(self.channels*16, self.channels*8, 3, 2, 1),
            nn.SELU(),
            # # conv_3
            nn.Conv2d(self.channels*8, self.channels*4, 3, 2, 1),
            nn.SELU(),
            # # conv_4
            nn.Conv2d(self.channels*4, self.channels*2, 3, 2, 1),
            nn.SELU(),
            # # exit_conv
            nn.Conv2d(self.channels*2, self.channels, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(128, 96), nn.SELU(), nn.Dropout(0.10),
            nn.Linear(96, 64)
        )

        # decoder model
        self.int_dec = nn.Sequential(
            nn.Linear(64, 96), nn.SELU(), nn.Dropout(0.10),
            nn.Linear(96, 128)
        )

        self.decoder = nn.Sequential(
            # conv_1
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=2, stride=1, padding=1, bias=True),
            nn.SELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # conv_2
            nn.Conv2d(self.channels*2, self.channels*4, 2, 1, 0),
            nn.SELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # # conv_3
            nn.Conv2d(self.channels*4, self.channels*8, 2, 1, 0),
            nn.SELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # # conv_4
            nn.Conv2d(self.channels*8, self.channels*16, 2, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.SELU(),
            # # exit_conv
            nn.Conv2d(self.channels*16, self.in_size[0], 3, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img):
        z = self.encoder(img)
        x = self.int_dec(z)
        x = x.view(z.size(0), 8, 4, 4)
        img_hat = self.decoder(x)
        return img_hat

# AE = AE(in_size)
# _ = AE(normed_maps[:100])
# print(_.shape)
#%%
class LitImgAE(pl.LightningModule):
    def __init__(self, in_size, lr=2e-4):
        super().__init__()
        self.in_size = in_size
        self.lr = lr
        self.mse = pl.metrics.regression.MeanSquaredError()
        self.ssim = pl.metrics.regression.SSIM()

        # enhacer model
        self.ae = AE(self.in_size)
        self.ae.apply(self.weights_init)

    def forward(self, maps):
        maps_hr_out= self.ae(maps)
        return maps_hr_out

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x, _ = batch
        pred_x = self(x)
        train_loss = self.mse(pred_x, x)

        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def _shared_eval(self, batch, batch_idx, prefix):
        torch.cuda.empty_cache()
        x, _ = batch
        pred_x = self(x)
        loss = F.mse_loss(pred_x, x)

        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return loss
        # return total_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), self.lr)
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=8, verbose=True) #0.5 works
        # return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)

#%%
#
X = int(len(dataset) * 0.8)
Y = int(len(dataset) - X)
Z = 0
split = [X, Y, Z]

# data model
dm = DataModule(dataset, batch_size, split, seed)

# %%
if torch.cuda.is_available():
    precision = 16
    gpu = 1
else:
    precision = 32
    gpu = 0

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, split, generator=torch.Generator().manual_seed(seed))

early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=50) #10 for mine
m_ae_model = LitImgAE(in_size, lr)
summary(m_ae_model, in_size)

CHECK = False

if CHECK:
    trainer = pl.Trainer(fast_dev_run=True)
else:
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=50, stochastic_weight_avg=True,
                         callbacks=[early_stopping], precision=precision, gpus=gpu)

# load entire dataset for training
# ds_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# trainer.fit(m_ae_model, datamodule=dm)
# torch.save(m_ae_model.state_dict(), "m_ae_model")
# %%
def data_access(folder=None):
    if folder == 'train':
        x, _ = train_dataset[:]
    elif folder == 'val':
        x, _ = valid_dataset[:]
    elif folder == 'test':
        x, _ = test_dataset[:]
    else:
        ValueError('Wrong loading')

    return x

folder = "val"  # "train", "val", "test"

im = data_access(folder)
# %%
m_ae_model.load_state_dict(torch.load('m_ae_model'))

im = normed_maps

m_ae_model.eval()
with torch.no_grad():
    im_hat = m_ae_model(im)
 
im, im_hat = im.numpy(), im_hat.numpy()

# %%
def plots(im, im_hat):
    for i in range(25):
        plt.figure(2, figsize=(7, 7))
        plt.subplot(5, 5, i+1)
        plt.imshow(im[i, 0], interpolation='none', cmap='jet')
        plt.clim(im.min(), im.max())
        plt.axis('off')
        plt.tight_layout()

        plt.figure(3, figsize=(7, 7))
        plt.subplot(5, 5, i+1)
        plt.imshow(im_hat[i, 0], interpolation='none', cmap='jet')
        plt.clim(im.min(), im.max())
        plt.axis('off')
        plt.tight_layout()

        # plt.figure(4, figsize=(7, 7))
        # plt.subplot(5, 5, i+1)
        # plt.imshow(im[i, 0]-im_hat[i, 0], interpolation='none', cmap='jet')
        # plt.clim(im.min(), im.max())
        # plt.axis('off')
        # plt.tight_layout()

    plt.show()
    plt.close(1); plt.close(2); plt.close(3);# plt.close(4)

def dists(im, im_hat):
    sns.distplot(im.flatten(), bins=100, label='true')
    sns.distplot(im_hat.flatten(), bins=100, label='pred')
    plt.legend()
    plt.show()

plots(im, im_hat)

for i in range(9):
    plt.figure(44, figsize=(8, 8))
    plt.subplot(3, 3, i+1)
    plt.imshow(im_hat[i, 0], interpolation='none', cmap='jet')
    plt.suptitle('Pred Maps')
    plt.axis('off')

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
