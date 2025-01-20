import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
import pdb
import torch.optim as optim
from torchsummary import summary
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from torch import linalg as LA
from oi_utils import data_reader
import argparse
from torch.autograd import Variable




parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=4000, help="number of epochs of training")
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
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.get_device_name())
device = torch.device('mps')

def load_pretrained_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('mps')), strict=False)

def compute_vis_matrix(im, uu, vv, scale,):
    sz = im.shape
    if sz[2] % 2 == 0:
        y, x = torch.meshgrid(torch.linspace(-torch.floor(torch.tensor(sz[2] / 2 - 1)), torch.floor(torch.tensor(sz[2] / 2)), sz[2]),
                              torch.linspace(torch.floor(torch.tensor(sz[3] / 2 - 1)), -torch.floor(torch.tensor(sz[3] / 2)), sz[3]))
    else:
        y, x = torch.meshgrid(torch.linspace(-torch.floor(torch.tensor(sz[2] / 2)), torch.floor(torch.tensor(sz[2] / 2)), sz[2]),
                              torch.linspace(torch.floor(torch.tensor(sz[3] / 2)), -torch.floor(torch.tensor(sz[3] / 2)), sz[3]))

    x_temp = x * scale
    y_temp = y * scale
    xx = x_temp.reshape(-1).to('mps')
    yy = y_temp.reshape(-1).to('mps')
    im_temp = im.reshape(-1)
    
    #u_tensor = torch.tensor(u, dtype=torch.float32)
    #v_tensor = torch.tensor(v, dtype=torch.float32)

    arg = -2.0 * np.pi * (uu * yy + vv * xx)
    reales = torch.dot(im_temp, torch.cos(arg.to('mps')))
    imaginarios = torch.dot(im_temp, torch.sin(arg.to('mps')))
    
    complex_vis = torch.complex(reales, imaginarios)
    
    #visib = complex_vis.abs()
    #phase = complex_vis.angle()
    return complex_vis
    
def MOD360(x):
    rr = (((x +180) % 360)+360) % 360 - 180
    return rr

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
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
    
def new_loss(discrim, inputs, scale,  data_dict):
    vis_data = torch.tensor(data_dict['vis'], dtype=torch.float32).to('mps')
    vis_data_err = torch.tensor(data_dict['vsigma'], dtype=torch.float32).to('mps')
    cphase_data = torch.tensor(data_dict['cphase'], dtype=torch.float32).to('mps')
    cphase_data_err = torch.tensor(data_dict['sigmacp'], dtype=torch.float32).to('mps')
    ucoord = torch.tensor(data_dict['u']).to('mps', dtype=torch.float32)
    vcoord = torch.tensor(data_dict['v']).to('mps', dtype=torch.float32)
    index_cp = torch.tensor(data_dict['index_cp'], dtype=torch.float32).to('mps')
    visib_t = torch.zeros([len(ucoord)], dtype=torch.complex64)

    for mm in range(len(ucoord)):
            visib_t[mm] = compute_vis_matrix(inputs, ucoord[mm], vcoord[mm], scale)
    
    #### Compute the closure phases ########
    cphase_t = torch.zeros([len(cphase_data)], dtype=torch.complex64)
    for ll in range(len(cphase_data)):
            cphase_t[ll] = visib_t[index_cp[ll, 1].int()] * visib_t[index_cp[ll, 2].int()] * torch.conj(visib_t[index_cp[ll, 3].int()])
    
    cphase_model = torch.rad2deg((cphase_t.angle() + np.pi) % (2 * np.pi) - np.pi)
    #pdb.set_trace()
    im_fake = torch.sum((visib_t.abs().to('mps') - vis_data)**2 / (vis_data_err)**2)
    mse_cphase = torch.sum((MOD360(cphase_model.to('mps') - cphase_data))**2 / cphase_data_err**2)
    dtype = torch.float
    #print(im_fake /(len(vis_data)), mse_cphase / len(cphase_data))
    #print(im_fake /(len(vis_data)), mse_cphase / len(cphase_data), torch.abs(SSIM_f))

    return im_fake /(len(vis_data)) +mse_cphase / len(cphase_data) - 10. * torch.log10(discrim[0][0])
    #+ 0.5*LA.norm(torch.squeeze(inputs), ord=0)

    
def main():
    epochs = 1000
    scale = 0.004 / 1000 / 3600 * np.pi / 180
    pretrained_weightsG_path = 'model_GAN-generator64i_all_500.pth'
    pretrained_weightsD_path = 'model_GAN-discriminator64i_all_500.pth'
    data_file = 'SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits'
    
    dict_data = data_reader(data_file)
    
    train_loss = 0
    train_save = []
    
    device = torch.device("mps")
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    #Tensor =  torch.tensor(data, dtype=*, device='cuda') if device.type == 'cuda' else torch.FloatTensor
    
    
    for ww in range(3 ):   #### This loop is for the number of images to be created

        # Initialize generator and discriminator
        generator = Generator()
        discriminator = Discriminator()

        if device.type == 'mps':
            generator.to('mps')
            discriminator.to('mps')
            adversarial_loss.to('mps')

       
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(opt.b1, opt.b2))
        TMP1 = Generator() 
        summary(TMP1,  (1, 100), device='cpu')
        TMP2 = Discriminator()
        summary(TMP2,  (1, 64, 64), device='cpu')

        load_pretrained_weights(generator, pretrained_weightsG_path)
        load_pretrained_weights(discriminator, pretrained_weightsD_path)

        z = Variable(torch.tensor(np.random.normal(0., 1., (1, opt.latent_dim)), dtype=torch.float32, device='mps'))
        im_inits = generator(z)
        #for name, param in generator.named_parameters():
        #    print(name)
        pyfits.writeto('prueba1_init.fits', np.squeeze(im_inits.cpu().detach().numpy().astype(np.float32)), overwrite=True)
        
        for k in range(epochs):
        
            generator.zero_grad()
            TMP2 = generator(z) 
            
            with torch.no_grad():
                discriminator.zero_grad() # Start the optimizer for the DIscriminator 
                output_discriminator = discriminator(TMP2)
            
            loss = new_loss(output_discriminator, TMP2, scale,  dict_data)
            loss.backward()
            optimizer_G.step()
        
            print(ww, k, output_discriminator, loss.item())
        
        
        if ww == 0:
            im_cube = np.squeeze(TMP2.cpu().detach().numpy())
        else:
            im_cube = np.dstack((im_cube, np.squeeze(TMP2.cpu().detach().numpy())))
        
        #pyfits.writeto('prueba1.fits', np.squeeze(TMP.cpu().detach().numpy()), overwrite=True)
    
    
        ucoord = torch.tensor(dict_data['u'])
        vcoord = torch.tensor(dict_data['v'])
        vis_data = torch.tensor(dict_data['vis'])
        vis_data_err = torch.tensor(dict_data['vsigma'])
    
        visib_im = torch.zeros([len(ucoord)], dtype=torch.complex64) 
        for mm in range(len(ucoord)):
            visib_im[mm] = compute_vis_matrix(TMP2, ucoord[mm], vcoord[mm], scale)
    
    
        fig1, (ax1, ax2) = plt.subplots(1,2)
        ax1.errorbar(np.sqrt(ucoord**2 + vcoord**2), np.abs(vis_data), yerr=vis_data_err, fmt='o', color='black')
        ax1.plot(np.sqrt(ucoord**2 + vcoord**2), np.abs(visib_im.detach().numpy()), 'o', color='red', zorder=1000)
    
        u1 = torch.tensor(dict_data['u1'])
        v1 = torch.tensor(dict_data['v1'])
        cphase_data = torch.tensor(dict_data['cphase'])
        cphase_data_err = torch.tensor(dict_data['sigmacp'])
        index_cp = torch.tensor(dict_data['index_cp'])
    
        cphase_im = torch.zeros([len(cphase_data)], dtype=torch.complex64)
        for ll in range(len(cphase_data)):
                cphase_im[ll] = visib_im[index_cp[ll, 1].int()] * visib_im[index_cp[ll, 2].int()] * torch.conj(visib_im[index_cp[ll, 3].int()])
    
        cphase_im = torch.rad2deg((cphase_im.angle() + np.pi) % (2 * np.pi) - np.pi)
    
        ax2.errorbar(np.sqrt(u1**2 + v1**2), cphase_data, yerr=cphase_data_err, fmt='o', color='black')
        ax2.plot(np.sqrt(u1**2 + v1**2), cphase_im.detach().numpy(), 'o', color='red', zorder=1000)
    
    
    im_mean = np.mean(im_cube, axis=2)
    im_median = np.median(im_cube, axis=2)
    pyfits.writeto('im_median128_e1000_f500.fits', im_median, overwrite=True)
    pyfits.writeto('im_mean128_e1000_f500.fits', im_mean, overwrite=True)
    im_temp = np.swapaxes(im_cube, 2,1)
    im_temp = np.swapaxes(im_temp, 1,0)
    pyfits.writeto('im_cube128_e1000_f500.fits',  im_temp, overwrite=True)
    
    plt.show()
    pdb.set_trace()


if __name__ == "__main__":
    main()
