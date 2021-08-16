#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import torch.nn as nn
import torch.functional as F
import random
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
import collections

sns.set_style("whitegrid", {'axes.grid': False})

# setting seed NO. if necessary
manualSeed = 999
print('Random Seed:', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
pl.seed_everything(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.determinstic = True

# %%
# setting directory to the data
def data_load(field=None, current_path=None):
    # return maps, data

    if field == "Gaussian":
        path = os.path.join(current_path, "Gaussian")
        maps_file = "\working_Gauss.npy"
        data_file = "\working_gaussian_data.npy"
    elif field == "Channel":
        path = os.path.join(current_path, "Channel")
        maps_file = "\channel_maps.npy"
        data_file = "\channel_data.npy"
    else:
        raise ValueError("Wrong field requested")

    maps_orig = np.load(os.path.join(path+maps_file))
    maps = np.reshape(maps_orig.T, (-1, 1, 100, 100))
    data = np.load(os.path.join(path+data_file))

    # returns maps(NxWxH)
    return maps, data

# Choose the field
current_path = os.getcwd() + r'\Models and Files'
field = "Gaussian" #"Channel"
# maps, data = data_load(field, current_path)

SLICE = 2000
MAPS = np.load('channels_uncond_10k.npy').T
MAPS = MAPS[:SLICE, :]
maps = MAPS.reshape(-1, 1, 64, 64)

print("Maps dims: ", maps.shape)

# %%
# GENERATOR
class Generator(nn.Module):
    def __init__(self, latent_dim=64, img_shape=None):
        super().__init__()
        self.img_shape = img_shape
        self.init_size = 8 #self.img_shape[1] // 4

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 64*self.init_size**2), nn.LeakyReLU(0.2, inplace=True))
        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=8, out_channels=img_shape[0], kernel_size=3, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
        )
    
    def forward(self, z):
        out = self.l1(z)
        # import pdb; pdb.set_trace()
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

#DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # The height and width of downsampled image
        #
        ds_size = 4

        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size, 1))
    def forward(self, img):
        out = self.disc(img)
        # import pdb; pdb.set_trace()
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity       

#%%

# import pdb; pdb.set_trace() for debugging
# generator check up
# z_dim = 50#128
# val_z = torch.randn(8, z_dim)
# img_shape = (1, 64, 64)
# gen = Generator(z_dim, img_shape)
# im_hat = gen(val_z)

# # discriminator check
# sample = torch.randn((3, 100, 100))
# disc = Discriminator(img_shape)
# y_hat = disc(im_hat.detach())

# %%
# DCGAN
class DCGAN(pl.LightningModule):
    def __init__(self, latent_dim=128, lr=0.0002, b1=0.9, b2=0.999, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.k_step = 0

        ### initializing networks
        img_shape = (1, 64, 64)
        self.generator = Generator(self.latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)

        # application of weight
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        #
        # for sampling
        self.validation_z = None
        self.imgs_ref = None
        self.logging_dict = {"g_loss": [], "g_avd_loss": [],
                             "d_loss": [], "d_avd_loss": [],
                             }

    def forward(self, z):
        return self.generator(z)

    ### weight initialization
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        
    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        self.validation_z = z
        # to ensure same set to be generated
        if self.k_step == 0:
            # self.pcs_sample = pcs
            self.imgs_ref = imgs

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_images = self(z)

            # log sampled images
            sample_imgs = self.generated_images[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth results (-> all fake)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log('g_loss', g_loss, prog_bar=True)
            output = g_loss

            # dictionary logging
            self.logging_dict['g_loss'].append(g_loss)
           
            return output
        
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            
            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of the two losses
            d_loss = (real_loss + fake_loss)
            self.log('d_loss', d_loss, prog_bar=True)
            output = d_loss
            self.logging_dict['d_loss'].append(d_loss)

            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        # image sampling and saving
        if self.k_step == 0 or self.k_step % 24 == 7:
            image_set = sample_imgs[:9, 0].cpu().detach().numpy()
            plt.figure(777, figsize=(7, 7))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(image_set[i], interpolation='nearest', cmap='jet')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'Images\\samples_{self.k_step}.png')
            plt.close(777)

        if self.k_step == 0:
            image_ref_set = self.imgs_ref[:9, 0].cpu().detach().numpy()
            plt.figure(888, figsize=(7, 7))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(image_ref_set[i], interpolation='nearest', cmap='jet')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(r'Images\\Ref_samples.png')
            plt.close(888)

        self.k_step += 1

# %%
# defining the hyperparameters
n_epochs = 600
z_dim = 64
batch_size = 100
lr = 0.0002

# dataloader and transforms
maps_max = np.abs(maps).max()
maps_normed = maps / maps_max  # -1 to 1 for tanh() in the end of Conv's
maps_normed = torch.from_numpy(maps_normed).float() # setting to the same type to match weights

# dataloader = DataLoader(maps_normed, batch_size=batch_size, shuffle=True)

# RUN THE MODEL 
# TRAIN
if torch.cuda.is_available():
    precision = 16
    gpu = 1
    dataloader = torch.utils.data.DataLoader(maps_normed, batch_size=batch_size, pin_memory=True, shuffle=True)
else:
    precision = 32
    gpu = 0
    dataloader = torch.utils.data.DataLoader(maps_normed, batch_size=batch_size, pin_memory=False, shuffle=True)

dc_gan = DCGAN(z_dim, lr, batch_size=batch_size)

CHECK = False

if CHECK:
    trainer = pl.Trainer(fast_dev_run=True)
else:
    trainer = pl.Trainer(max_epochs=n_epochs, progress_bar_refresh_rate=50, stochastic_weight_avg=False,
                         precision=precision, gpus=gpu)
# running the model
# trainer.fit(dc_gan, dataloader)

# %%
# enable to initiate tensorboard in the terminal
# tensorboard --log_dir lightning_logs/
# torch.save(dc_gan.state_dict(), "gan_small")
dc_gan.load_state_dict(torch.load('gan_small'))

#%%
z = torch.randn(25, z_dim)
dc_gan.eval()
imgs = dc_gan(z).detach().numpy()

plt.figure(figsize=(7, 7))
for i in range(imgs.shape[0]):
    _ = imgs
    
    plt.subplot(5, 5, i+1)
    plt.imshow(_[i, 0], interpolation='nearest', cmap='jet')
    # plt.colorbar()
    plt.axis('off')
plt.tight_layout()
plt.show()
