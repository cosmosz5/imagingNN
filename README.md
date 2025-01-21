# imagingNN
This is a repository with code to perform interferometric image reconstruction using Neural Networks


To download the .npz cube with the simulated black hole images visit: https://drive.google.com/drive/folders/1pTl6s18CUovAfo4ZU1eArF1bhI1kx0-p?usp=share_link 

The current code in the repository ONLY runs on Apple computers with M processors. To run the GAN trainning, download the repository and type the following command in the Terminal: 

```
python GAN2_EHT2.py
```
This will go over 500 iterations to train the Generator and Discriminator nets. Every 50 iterations, the code will create a sample image (in .fits format). This is to check the evolution of the trainning process. 

After the trainning process is finished, to optimize the GAN to the given interferometric data run the following command in the Terminal: 

```
python optimize_bh5.py
```
Currently, the repository contains trainned weights for the GAN (model_GAN-generator64i_all_500.pth, model_GAN-discriminator64i_all_500.pth). So, it is not necessary to run all the trainning process. Unless changes to the default setup of the GAN are made. 
