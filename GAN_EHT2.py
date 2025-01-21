import torch
import torch.nn as nn
from IPython.display import Image
import torchvision
from torchview import draw_graph
from torchsummary import summary
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import astropy.io.fits as pyfits
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import pdb
from resnet_utils import Bottleneck, ResNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
device = torch.device('mps')


def load_pretrained_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path), strict=False)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(128, 128, 3, stride=1, padding=1),
            #nn.BatchNorm2d(128, 0.8),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(128, 64, 3, stride=1, padding=1),
            #nn.BatchNorm2d(64, 0.8),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            
            nn.Upsample(scale_factor = 4, mode = 'bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size = 4, stride= 2),
            #nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 18, 18)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor = 4, mode = 'bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size = 4, stride= 2),
            #nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2), # (32, 36, 36)
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size = 5, stride= 1, padding='same'),
            #nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (1, 70, 70)
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if device.type == 'mps':
    generator.to('mps')
    discriminator.to('mps')
    adversarial_loss.to('mps')

#pretrained_weightsG_path = 'model_GAN-generator64i_2650.pth'
#pretrained_weightsD_path = 'model_GAN-discriminator64i_2650.pth'
#load_pretrained_weights(generator, pretrained_weightsG_path)
#load_pretrained_weights(discriminator, pretrained_weightsD_path)


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(opt.b1, opt.b2))

#Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

Tensor = torch.FloatTensor


def train(optimizer_G, optimizer_D, train_loader,epoch):

    d_loss_tot = 0
    d_loss_save = []

    g_loss_tot = 0
    g_loss_save = []
    
    for batch_idx, data in enumerate(train_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(data[0].size(0), 1).fill_(1.0), requires_grad=False).to('mps')
        fake = Variable(Tensor(data[0].size(0), 1).fill_(0.0), requires_grad=False).to('mps')

        # Configure input
        real_imgs = Variable(Tensor(data[0])).to('mps')

        generator.zero_grad() ##Start the optimizer for the Generator

        # Sample noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (data[0].shape[0], opt.latent_dim)))).to('mps')

        # Generate a batch of images
        gen_imgs = generator(z)

        all_samples = torch.cat((real_imgs, gen_imgs))
        all_samples_labels = torch.cat((valid, fake))

        #DIscriminator trainning 
        discriminator.zero_grad() # Start the optimizer for the DIscriminator 
        output_discriminator = discriminator(all_samples)

        d_loss = adversarial_loss(output_discriminator, all_samples_labels)
        d_loss.backward()
        optimizer_D.step()

        #Data for Generator trainning 
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (data[0].shape[0], opt.latent_dim)))).to('mps')
        generator.zero_grad()
        gen_ims = generator(z)
        output_discriminator_generated = discriminator(gen_ims)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(output_discriminator_generated, valid)
        g_loss.backward()
        optimizer_G.step() #Give one step to the optimizer for the Generator

        # Measure discriminator's ability to classify real from generated samples
        #real_loss = adversarial_loss(discriminator(real_imgs), valid)
        #fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        #d_loss = (real_loss + fake_loss) / 2

        #d_loss.backward() ##Backpropagation!!!
        #optimizer_D.step()


        d_loss_tot += d_loss.item()
        g_loss_tot += g_loss.item()
        if batch_idx % 100 == 0:
            
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" \
                % (epoch, opt.n_epochs, batch_idx, len(train_loader), d_loss.item(), g_loss.item()))

    if (epoch % 50 == 0):
        torch.save(generator.state_dict(), f'model_GAN-generator64i_all_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'model_GAN-discriminator64i_all_{epoch}.pth')
        with torch.no_grad():
            z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
            aa = generator(z)
            pyfits.writeto('rec64_GAN_all_'+str(epoch)+'.fits', np.squeeze(aa.cpu().detach().numpy()), overwrite=True)
        print(f'Checkpoint saved at epoch {epoch}')

    if epoch == 1:
        d_loss_save = d_loss_tot / len(train_loader.dataset)
        g_loss_save = g_loss_tot / len(train_loader.dataset)
    else:
        d_loss_save = np.append(d_loss_save, d_loss_tot / len(train_loader.dataset))
        g_loss_save = np.append(g_loss_save, g_loss_tot / len(train_loader.dataset))

    print(f'====> Epoch: {epoch} Average loss: {np.mean(d_loss_save):.4f}')
    print(f'====> Epoch: {epoch} Average loss: {np.mean(g_loss_save):.4f}')

    return g_loss_save, d_loss_save


def main():
    if device.type == 'cuda':
        print(torch.cuda.get_device_name())
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    epochs = opt.n_epochs
    batch_size = opt.batch_size
	#latent_dim = 1024
    #learning_rate = 5e-5
    # Load your dataset here
    load_data = np.load('ims_bhole_4uas_all_threshold.npz')
    train_images = load_data['data']
    X_train = train_images.reshape(1, 64,64, 120000)
    #X_train = X_train[:, :, :, 100000::]
    X_train = np.swapaxes(X_train, 3, 2)
    X_train = np.swapaxes(X_train, 2, 1)
    X_train = np.swapaxes(X_train, 1, 0)

    train_size = int(0.8 * len(X_train))
    test_size = len(X_train) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(X_train, [train_size, test_size])
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_dataset, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_dataset, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ##Print the network architecture ######
    z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
    TMP1 = Generator() 
    summary(TMP1,  (1, 100), device='cpu')
    TMP2 = Discriminator()
    summary(TMP2,  (1, 64, 64), device='cpu')

    for epoch in range(1, epochs + 1):
        g_loss_save, d_loss_save = train(optimizer_G, optimizer_D, train_loader, epoch)
        np.savez('train_losses64_GAN_all.npz', g_loss=g_loss_save, d_loss=d_loss_save)

if __name__ == "__main__":
    main()
