from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import hflip,adjust_brightness,adjust_contrast,adjust_hue,adjust_saturation
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import random
from tqdm import tqdm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from datetime  import datetime

#https://github.com/1adrianb/face-alignment

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

parser.add_argument('--schedular', default='False')
parser.add_argument('--log', default='False')
parser.add_argument('--load_epoch', default='')
parser.add_argument('--load_plot_file', default='')
parser.add_argument('--start_over', default='False')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    # folder dataset

def get_label(name): #converts names to int labels
    if "ANG" in name:
        return 0
    elif "HAP" in name:
        return 1
    elif "NEU" in name:
        return 2
    elif "SAD" in name:
        return 3
    elif "DIS" in name:
        return 4
    elif "FEA" in name:
        return 5

class Audio_video_dataset(Dataset):


    def __init__(self, h5_file, video_root_dir, transform=None):


        self.h5_file = h5_file

        self.audio_dict = {}

        self.frame_list = []

        self.frame_num = 0

        with  h5py.File(self.h5_file, 'r') as f:
            self.keys = list(f.keys())
            for key in tqdm(self.keys):
                self.audio_dict[key]=f[key][:]
                #self.frame_list = self.frame_list + [key+"/"+str(x) for x in range(2,self.audio_dict[key].shape[0]+1)] #Start from 2 since 1 is reference, add 1 to the endbcause of the lenght
        self.video_root_dir = video_root_dir
        self.transform = transform

        """
        random.shuffle(self.keys)

        self.keys = self.keys[0:10]


        self.img_dict = {}
        for i in tqdm(self.keys):
            self.img_dict[i] = self.read_images(os.path.join(self.video_root_dir,i),self.audio_dict[i].shape[0],False)
            self.frame_num+=self.img_dict[i].size(0)
        """

    def __len__(self):
        return len(self.keys)


    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py
    def get_frame_num(self):
        return self.frame_num

    def read_images(self, video_dir,frame_num,flip):
        image_list=os.listdir(video_dir)

        X = []
        brightness = torch.rand(1)*0.9+0.4
        contrast = torch.rand(1)+0.5
        saturation = torch.rand(1)*1.5
        #hue = torch.rand(1)-0.5/5

        for i in range(1,frame_num+1):
            image = Image.open(os.path.join(video_dir,'image-{:d}.jpeg'.format(i)))

            if self.transform:

                if flip==True:
                    image = hflip(image)
                """
                image = adjust_brightness(image,brightness_factor=brightness) #0-2
                image = adjust_contrast(image,contrast_factor=contrast) #0-2
                image = adjust_saturation(image,saturation_factor=saturation)
                #image  = adjust_hue(image,hue_factor = hue )#-0.5,0.5
                """
                image = self.transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.keys[idx] #current elemen (audio,video)

        audio_data = torch.from_numpy(self.audio_dict[selected_elem])
        frame_num = audio_data.size(0)

        label = get_label(selected_elem)

        sequence_name = os.path.join(self.video_root_dir,selected_elem)

        flip = torch.rand(1).item()>0.5
        #print("flip",flip)
        #flip = flip>0.5

        image_y = self.read_images(sequence_name,frame_num,flip)
        """
        image_y = self.img_dict[selected_elem]
        """

        #rand_frame = (torch.LongTensor(1).random_(0, frame_num)+1).item()
        #print("image_y shape",image_y.shape)
        ref_image = image_y[0,:,:,:]


        return ref_image,image_y,audio_data,label

dataset_train = Audio_video_dataset(h5_file="../data/audio/6class_indep/audio_features_6class(1pad)_train.hdf5",
                                           video_root_dir='../data/video/cropped_face_frames2',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
dataset_test = Audio_video_dataset(h5_file="../data/audio/6class_indep/audio_features_6class(1pad)_test.hdf5",
                                           video_root_dir='../data/video/cropped_face_frames2',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))



batch_size = 1


def customBatchBuilder(samples):
    ref_image,image_y,audio_data,label = zip(*samples)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)
    image_y = pad_sequence(image_y, batch_first=True, padding_value=0)

    ref_img_tensor = torch.Tensor(ref_image[0]).unsqueeze(0)
    for i in range(1,len(ref_image)):
        ref_img_tensor=torch.cat((ref_img_tensor,torch.Tensor(ref_image[i]).unsqueeze(0)),axis=0)


    #image_seq= torch.Tensor(image_seq[0])
    #image_seq = image_seq.view((-1,3,64,64))

    label = torch.Tensor(label).long()
    return ref_img_tensor,image_y,audio_data,label



train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)




test_loader = DataLoader(dataset_test, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)



train_set_size = len(dataset_train)
test_set_size = len(dataset_test)

print("Train set size:",train_set_size)
print("Test set size:",test_set_size)

print("train_loader",len(train_loader))
print("test_loader",len(test_loader))


nc=3

#device = torch.device("cpu")
device = torch.device("cuda")# if opt.cuda else "cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628

class Generator(nn.Module):
    def __init__(self,img_latent_dim,aud_latent_dim,emo_latent_dim):
        super(Generator, self).__init__()
        self.img_latent_dim = img_latent_dim
        self.aud_latent_dim = aud_latent_dim
        self.emo_latent_dim = emo_latent_dim

        self.ndf = 64
        self.ngf = 64
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.img_enc_l5 = nn.Conv2d(self.ndf * 4, self.ndf*8, 4, 1, 0, bias=False)
        self.img_linear = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(self.ndf * 8,self.ndf * 2))


        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, self.ndf*2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf* 2, self.ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
        )



        self.lstm_hiddend_dim = self.aud_latent_dim*2
        self.audio_lstm_linear = nn.Linear(self.lstm_hiddend_dim,self.aud_latent_dim)

        self.encoded_dim = self.img_latent_dim + self.aud_latent_dim + self.emo_latent_dim + 128

        self.linear = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(self.encoded_dim,self.encoded_dim),nn.LeakyReLU(0.2, inplace=True))


        self.decoder_l1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.ndf*16 , self.ngf * 4, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(self.ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder_l2 = nn.Sequential(    # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d( self.ngf * 8,  self.ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder_l3 = nn.Sequential(    # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 8,  self.ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder_l4 = nn.Sequential(    # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 4, self.ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder_l5 = nn.Sequential(    # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf*2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.label_emb = nn.Embedding(6, self.emo_latent_dim)

        self.lstm = nn.LSTM(self.aud_latent_dim, self.lstm_hiddend_dim, 1, batch_first=True)
        self.audio_linear = nn.Linear(self.ndf*60*3,self.aud_latent_dim)

        self.noise_lstm = nn.LSTM(128,128,1, batch_first=True)
        self.emo_lstm = nn.LSTM(self.emo_latent_dim,self.emo_latent_dim,1, batch_first=True)

    def forward(self,ref_image,audio_data,noise,emo_label):

        batch_size = ref_image.size(0)
        step_size = audio_data.size(1)
        #print("audio size",audio_data.shape)
        enc_l1 = self.img_enc_l1(ref_image)
        #print("enc1",enc_l1.shape)
        enc_l2 = self.img_enc_l2(enc_l1)
        #print("enc2",enc_l2.shape)
        enc_l3 = self.img_enc_l3(enc_l2)
        #print("enc3",enc_l3.shape)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print("enc4",enc_l4.shape)
        enc_l5 = self.img_enc_l5(enc_l4)
        #print("enc5",enc_l5.shape)
        #print("ref",x1.shape)
        #print("audio befoer",audio_data.shape)
        x2 = self.audio_encoder(audio_data.view((-1,1,60,3)))
        #print("audio",x2.shape)
        x2 = self.audio_linear(x2.view(-1,self.ndf*60*3))

        #print("audio linear",x2.shape)
        #print("one hot label",one_hot_label.shape)

        h0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)

        h1 = torch.zeros(1, batch_size, 128).to(device)
        c1 = torch.zeros(1, batch_size, 128).to(device)

        h2 = torch.zeros(1, batch_size, self.emo_latent_dim).to(device)
        c2 = torch.zeros(1, batch_size, self.emo_latent_dim).to(device)

        x, _ = self.lstm(x2.view((batch_size,-1,self.aud_latent_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.audio_lstm_linear(x.view(-1,self.lstm_hiddend_dim))

        embed = self.label_emb(emo_label).repeat(step_size,1,1,1)
        #print("embed dim",embed.shape)
        embed, _ = self.emo_lstm(embed.view((batch_size,-1,self.emo_latent_dim)),(h2,c2))

        y, _ = self.noise_lstm(noise.view((batch_size,-1,128)), (h1, c1))
        #print("x dim",x.shape)
        #print("embed dim",embed.shape)
        #print("y dim",y.shape)
        #print("enc_l5 dim",enc_l5.shape)

        img_linear = self.img_linear(enc_l5.view((-1,512)))

        z_latent = torch.cat((img_linear.view((1,-1,1,1)).repeat(step_size,1,1,1),
        x.view(-1,self.aud_latent_dim,1,1),embed.view(-1,self.emo_latent_dim,1,1),y.view(-1,128,1,1)),axis=1)
        #print("z_latent shape",z_latent.shape)
        z_latent = self.linear(z_latent.view((-1,self.encoded_dim)))

        dec_l1 = self.decoder_l1(torch.cat((z_latent.reshape((-1,self.encoded_dim,1,1)),enc_l5.repeat(step_size,1,1,1)),1))
        #print("dec1",dec_l1.shape)
        dec_l2 = self.decoder_l2(torch.cat((enc_l4.repeat(step_size,1,1,1),dec_l1),1))
        #print("dec2",dec_l2.shape)
        dec_l3 = self.decoder_l3(torch.cat((enc_l3.repeat(step_size,1,1,1),dec_l2),1))
        #print("dec3",dec_l3.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l4 = self.decoder_l4(torch.cat((enc_l2.repeat(step_size,1,1,1),dec_l3),1))
        #print("dec4",dec_l4.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l5 = self.decoder_l5(torch.cat((enc_l1.repeat(step_size,1,1,1),dec_l4),1))

        return dec_l5.view((batch_size,-1,3,64,64))

netG = Generator(128,128,128).to(device)
netG.apply(weights_init)


#netG.load_state_dict(torch.load("netG_epoch_149_continue2.pth"))

#torch.save(netG.main.state_dict(), 'netG_epoch_149_continue2.pt')

class Discriminator_img(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_img, self).__init__()
        self.ngpu = ngpu
        self.ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc*2, self.ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*4,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*8,affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:

            #F.dropout(input,p=0.8)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD_img = Discriminator_img(ngpu).to(device)
netD_img.apply(weights_init)
print(netD_img)




class Discriminator_vid(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator_vid, self).__init__()
        self.latent_dim = latent_dim
        self.ndf = 64
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*4,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, self.ndf*2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf * 2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf* 2, self.ndf, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf, 1, 1, 0, bias=False),
        )

        self.img_linear = nn.Linear(self.ndf * 2 * 4* 2,self.latent_dim)

        self.audio_linear = nn.Linear(self.ndf*60*3,self.latent_dim)

        self.encoded_dim = self.latent_dim*2

        self.lstm_hiddend_dim = self.encoded_dim*2

        self.lstm = nn.LSTM(self.encoded_dim, self.lstm_hiddend_dim, 1, batch_first=True)


        self.real_vid = nn.Sequential(nn.Linear(self.lstm_hiddend_dim,1),nn.Sigmoid())

    def forward(self,imgs,audio_data):


        step_size = audio_data.size(1)
        #print("audio size",audio_data.shape,step_size)
        enc_l1 = self.img_enc_l1(imgs.view((-1,3,32,64)))
        enc_l2 = self.img_enc_l2(enc_l1)
        enc_l3 = self.img_enc_l3(enc_l2)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print(enc_l4.shape)
        x1 = self.img_linear(enc_l4.view((-1,self.ndf * 2 * 4* 2)))
        #print(x1.shape)
        x2 = self.audio_encoder(audio_data.view((-1,1,60,3)))

        x2 = self.audio_linear(x2.view(-1,self.ndf*60*3))

        z_latent = torch.cat((x1.unsqueeze(2).unsqueeze(3),x2.view(-1,self.latent_dim,1,1)),axis=1)
        #print("z_latent size",z_latent.shape)

        h0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(z_latent.view((1,-1,self.encoded_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print("x size",x.shape)
        real_vid = self.real_vid(x[:,np.arange(0,step_size,5),:])

        return real_vid.squeeze(0).squeeze(1)




netD_vid = Discriminator_vid(128).to(device)
netD_vid.apply(weights_init)
print(netD_vid)

class Discriminator_emotion(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator_emotion, self).__init__()
        self.latent_dim = latent_dim
        self.ndf = 64

        self.cnn3d =  nn.Conv3d(3,1,(5,4,4),stride=(1,2,2),padding=(1,1,1))

        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, self.ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*4,affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.img_linear = nn.Linear(self.ndf * 2 * 2* 2,self.latent_dim)

        self.lstm_hiddend_dim = self.latent_dim*2

        self.lstm = nn.LSTM(self.latent_dim, self.lstm_hiddend_dim, 1, batch_first=True)

        self.lstm_linear= nn.Sequential(nn.LeakyReLU(0.2, inplace=False),
        nn.Linear(self.lstm_hiddend_dim,self.lstm_hiddend_dim))

        self.binary_classify =  nn.Sequential(nn.LeakyReLU(0.2, inplace=False),nn.Linear(self.lstm_hiddend_dim,1),nn.Sigmoid())
        self.class_classify =  nn.Sequential(nn.LeakyReLU(0.2, inplace=False),nn.Linear(self.lstm_hiddend_dim,6))

    def forward(self,imgs):

        #print("imgs shape",imgs.shape)
        cnn_3d = self.cnn3d(imgs.view(1,3,-1,64,64))
        #print("3dcnn",cnn_3d.shape)
        enc_l1 = self.img_enc_l1(cnn_3d.squeeze(0).view(-1,1,32,32))
        #print("enc_l1",enc_l1.shape)
        enc_l2 = self.img_enc_l2(enc_l1)
        #print("enc_l2",enc_l2.shape)
        enc_l3 = self.img_enc_l3(enc_l2)
        #print("enc_l3",enc_l3.shape)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print("enc_l4",enc_l4.shape)
        x1 = self.img_linear(enc_l4.view((-1,self.ndf * 2 * 2* 2)))
        #print(x1.shape)

        h0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(x1.view((1,-1,self.latent_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print("x shape",x.shape)
        x = self.lstm_linear(x[:,-1,:])
        #print("emo shape",emo.shape)
        binary_class = self.binary_classify(x)
        #print("binary_class shape",binary_class.shape)
        class_predict = self.class_classify(x)
        #print("class_predict shape",class_predict.shape)
        return binary_class.squeeze(0),class_predict



netD_emo = Discriminator_emotion(256).to(device)
netD_emo.apply(weights_init)
print(netD_emo)




"""
netD_seq_vid= Discriminator_seq_video(256).to(device)
netD_seq_vid.apply(weights_init)
print(netD_seq_vid)
"""


criterion = nn.BCELoss()
L1_loss = nn.L1Loss()
class_loss = nn.CrossEntropyLoss()

real_label = 1
fake_label = 0
#https://discuss.pytorch.org/t/loading-optimizer-dict-starts-training-from-initial-lr/36328/2
file_name = "lip_ganv8_unet_ac"
img_folder_dir = os.path.join("Img_folder",file_name)
if not os.path.exists(img_folder_dir):
    os.makedirs(img_folder_dir)
    os.makedirs(os.path.join(img_folder_dir,"test"))
    os.makedirs(os.path.join(img_folder_dir,"train"))

model_dir = os.path.join("Models",file_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_dir = os.listdir(os.path.join("Models",file_name))




now=datetime.now()

plot_file="./Img_folder/"+file_name+"/"+file_name+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"

# setup optimizer
optimizerD_img = optim.Adam(netD_img.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
optimizerD_vid = optim.Adam(netD_vid.parameters(), lr=0.00002, betas=(opt.beta1, 0.999))
optimizerD_emo = optim.Adam(netD_emo.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
#optimizerD_seq_vid = optim.Adam(netD_seq_vid.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))

start = 0

if (len(model_dir)!=0 or opt.load_epoch != '') and opt.start_over=='False':
    print("Loading old save")
    list_models = [int(i.split(".")[0].split("epoch_")[1]) for i in model_dir if "netD_img" in i]
    list_models.sort()
    print(list_models)
    load_epoch = list_models[-1]
    print(load_epoch)
    load_epoch=load_epoch

    netD_img.load_state_dict(torch.load('%s/Models/%s/netD_img_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    netD_vid.load_state_dict(torch.load('%s/Models/%s/netD_vid_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    netD_emo.load_state_dict(torch.load('%s/Models/%s/netD_emo_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    #netD_seq_vid.load_state_dict(torch.load('%s/Models/%s/netD_seq_vid_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    netG.load_state_dict(torch.load('%s/Models/%s/netG_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))

    ##########################################################################################################################
    optimizerD_img.load_state_dict(torch.load('%s/Models/%s/optimizerD_img.pth' % (opt.outf, file_name)))
    optimizerD_vid.load_state_dict(torch.load('%s/Models/%s/optimizerD_vid.pth' % (opt.outf, file_name)))
    optimizerD_emo.load_state_dict(torch.load('%s/Models/%s/optimizerD_emo.pth' % (opt.outf, file_name)))
    #optimizerD_seq_vid.load_state_dict(torch.load('%s/Models/%s/optimizerD_seq_vid.pth' % (opt.outf, file_name)))
    optimizerG.load_state_dict(torch.load('%s/Models/%s/optimizerG.pth' % (opt.outf, file_name)))

    start = int(load_epoch)+1

    #if opt.load_plot_file != '':
    plot_file_dir_list  = os.listdir(os.path.join("Img_folder",file_name))
    print(plot_file_dir_list)
    list_plots = [i for i in plot_file_dir_list if "csv" in i]
    print(list_plots)
    plot_file="./Img_folder/"+file_name+"/"+list_plots[0]



train_frame_num = dataset_train.get_frame_num()
test_frame_num =  dataset_test.get_frame_num()
if opt.schedular == "True":
    schedulerD_img = StepLR(optimizerD_img, step_size=15, gamma=0.1)
    schedulerD_vid = StepLR(optimizerD_vid, step_size=15, gamma=0.1)
    schedulerD_emo = StepLR(optimizerD_emo, step_size=15, gamma=0.1)
    #schedulerD_seq_vid = StepLR(optimizerD_seq_vid, step_size=15, gamma=0.1)
    schedulerG = StepLR(optimizerG, step_size=15, gamma=0.1)





print("Train frame num: ",train_frame_num)
print("Test frame num: ",test_frame_num)

print("train_real_emo_class_acc")

if not os.path.exists("./Img_folder/"+file_name):
    os.mkdir("./Img_folder/"+file_name)

if not os.path.exists("./Models/"+file_name):
    os.mkdir("./Models/"+file_name)




for epoch in range(start,start+opt.niter):

    train_l1_loss = 0
    train_d_img_loss = 0
    train_d_vid_loss = 0
    train_d_emo_loss = 0
    train_d_seq_vid_loss = 0

    train_real_img_acc= 0
    train_real_vid_acc = 0
    train_real_emo_acc = 0

    train_real_emo_class_acc = 0

    train_real_seq_vid_acc = 0

    train_fake_img_acc= 0
    train_fake_vid_acc = 0
    train_fake_emo_acc = 0
    train_fake_seq_vid_acc = 0


    train_g_img_loss = 0
    train_g_vid_loss = 0
    train_g_emo_loss = 0
    train_g_seq_vid_loss = 0
    train_gtotal_loss = 0
    train_dx = 0

    test_l1_loss = 0
    test_d_img_loss = 0
    test_d_vid_loss = 0
    test_d_emo_loss = 0
    test_d_seq_vid_loss = 0

    test_g_img_loss = 0
    test_g_vid_loss = 0
    test_g_emo_loss = 0
    test_g_seq_vid_loss = 0
    test_gtotal_loss = 0

    test_real_img_acc= 0
    test_real_vid_acc = 0
    test_real_emo_acc = 0
    test_real_emo_class_acc = 0
    test_real_seq_vid_acc = 0

    test_fake_img_acc= 0
    test_fake_vid_acc = 0
    test_fake_emo_acc = 0
    test_fake_seq_vid_acc = 0

    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        ref_image,image_y,audio_data, emo_label = data
        audio_data = audio_data.to(device)

        ################################################################
        # D_img fake
        ################################################################
        netD_img.zero_grad()
        real_cpu = image_y.to(device).view((-1,3,64,64))

        #print("ref image shape",ref_image.shape)
        #print("real_cpu image shape",real_cpu.shape)

        #real_cpu = augment(real_cpu)
        #print("augment real_cpu image shape",real_cpu.shape)
        batch_size_d = real_cpu.size(0)
        step_size = image_y.size(1)
        emo_label = emo_label.to(device)
        ref_image = ref_image.to(device)
        label = torch.full((batch_size_d//2,), real_label, device=device)
        #emo_labels =  torch.full((1,), emo_label.item(), device=device).long()

        #one_hot_label = torch.zeros(1,4).to(device) #number of class
        #one_hot_label[:,emo_label]=1

        t_vid=np.arange(0,step_size,5).shape[0]

        vid_label = torch.full((t_vid,), real_label, device=device)
        seq_label = torch.full((1,), real_label, device=device)
        #label[np.random.randint(batch_size,size=(batch_size//20))]=fake_label

        random_frames = torch.randperm(batch_size_d)[:batch_size_d//2] #torch.LongTensor(batch_size_d//2).random_(0, batch_size_d)
        #print("real cpu",real_cpu.shape)
        #print("real cpu2",real_cpu[random_frames,:,:,:].shape)

        #the line below is for reference concat
        output = netD_img(torch.cat((real_cpu[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1))
        #output = netD_img(real_cpu[random_frames,:,:,:])
        errD_img_real = criterion(output, label)
        errD_img_real.backward()
        train_real_img_acc+=(output.mean().item()>0.5)
        ################################################################
        # D_vid real
        ################################################################
        netD_vid.zero_grad()

        output_vid = netD_vid(real_cpu[:,:,32:64,:],audio_data)
        errD_real_vid = criterion(output_vid, vid_label)
        errD_real_vid.backward()
        train_real_vid_acc+=(output_vid.mean().item()>0.5)

        ################################################################
        # D_emo real
        ################################################################
        netD_emo.zero_grad()
        output_seq,class_pred = netD_emo(real_cpu)
        #print("seq label",seq_label.shape)
        errD_real_emo = criterion(output_seq, seq_label) + class_loss(class_pred,emo_label)
        errD_real_emo.backward()
        train_real_emo_acc+=(output_seq.item()>0.5)
        train_real_emo_class_acc+=(torch.argmax(class_pred) == emo_label).item()
        """
        ################################################################
        # D_seq_vid real
        ################################################################
        netD_seq_vid.zero_grad()
        output_seq = netD_seq_vid(real_cpu)
        #print("seq label",seq_label.shape)
        errD_real_seq_vid = criterion(output_seq, seq_label)
        errD_real_seq_vid.backward()
        train_real_seq_vid_acc+=(output_seq.item()>0.5)
        """
        ################################################################
        # D_img fake
        ################################################################
        # train with fake

        batch_size_g = image_y.size(0)

        #one_hot_label = torch.zeros(batch_size_g,4).to(device) #number of class
        #one_hot_label[range(batch_size_g),emo_label]=1


        noise = torch.randn(step_size,128, device=device)
        fake = netG(ref_image,audio_data,noise,emo_label)

        #fake = augment(fake.squeeze(0))
        fake = fake.squeeze(0)
        label.fill_(fake_label)
        vid_label.fill_(fake_label)
        seq_label.fill_(fake_label)
        #l1_loss = L1_loss(fake.squeeze(0)[:,:,32:64,:],real_cpu[:,:,32:64,:])*5

        #the line below is for reference concat
        output = netD_img((torch.cat((fake.view(-1,3,64,64).detach()[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1)))
        #output = netD_img(fake.view(-1,3,64,64).detach()[random_frames,:,:,:])
        errD_img_fake = criterion(output, label)
        errD_img_fake.backward()
        train_fake_img_acc+=(output.mean().item()<0.5)

        errD_img = errD_img_real + errD_img_fake
        if opt.schedular == "True":
            schedulerD_img.step()
        else:
            optimizerD_img.step()


        ################################################################
        # D_vid fake
        ################################################################

        output = netD_vid(fake.detach()[:,:,32:64,:],audio_data)
        errD_fake_vid = criterion(output, vid_label)
        errD_fake_vid.backward()
        train_fake_vid_acc+=(output.mean().item()<0.5)
        if opt.schedular == "True":
            schedulerD_vid.step()
        else:
            optimizerD_vid.step()

        ################################################################
        # D_emo fake
        ################################################################
        output_seq,class_pred = netD_emo(fake.detach())
        errD_fake_emo = criterion(output_seq, seq_label) + class_loss(class_pred,emo_label)
        errD_fake_emo.backward()
        train_fake_emo_acc+=(output_seq.item()<0.5)

        if opt.schedular == "True":
            schedulerD_emo.step()
        else:
            optimizerD_emo.step()
        """
        ################################################################
        # D_seq_vid fake
        ################################################################
        output_seq = netD_seq_vid(fake.detach())
        errD_fake_seq_vid = criterion(output_seq, seq_label)
        errD_fake_seq_vid.backward()
        train_fake_seq_vid_acc+=(output_seq.item()<0.5)
        if opt.schedular == "True":
            schedulerD_seq_vid.step()
        else:
            optimizerD_seq_vid.step()

        """
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        vid_label.fill_(real_label)
        seq_label.fill_(real_label)

        #the line below is for reference concat
        output = netD_img((torch.cat((fake.view(-1,3,64,64)[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1)))
        #output = netD_img(fake.view(-1,3,64,64)[random_frames,:,:,:])
        errG_img = criterion(output, label)

        output = netD_vid(fake[:,:,32:64,:],audio_data)
        errG_vid =  criterion(output, vid_label)

        output,class_pred = netD_emo(fake)
        errG_emo =  criterion(output, seq_label) + class_loss(class_pred,emo_label)


        errG_emo =  criterion(output, seq_label)
        """
        output = netD_seq_vid(fake)
        errG_seq_vid =  criterion(output, seq_label)
        """


        errG= errG_img +errG_vid + errG_emo #errG_seq_vid

        errG.backward()
        D_G_z2 = output.mean().item()
        if opt.schedular == "True":
            schedulerG.step()
        else:
            optimizerG.step()



        """
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        """

        train_d_img_loss += errD_img.item()
        train_d_vid_loss += (errD_fake_vid+errD_real_vid).item()
        train_d_emo_loss += (errD_fake_emo+errD_real_emo).item()
        #train_d_seq_vid_loss += (errD_fake_seq_vid+errD_real_seq_vid).item()


        train_g_img_loss += errG_img.item()
        train_g_vid_loss += errG_vid.item()
        train_gtotal_loss += errG.item()
        train_g_emo_loss += errG_emo.item()
        #train_g_seq_vid_loss += errG_seq_vid.item()

        """
        for name, p in netD_emo.named_parameters(): #,model._all_weights[0]): #prints gradients below
            #if n[:6] == 'weight':

            #print('===========\ngradient:{}\n----------\n{}----------\n{}'.format(p.grad,p.grad.shape,abs(p.grad).mean()))
            print("name:",name)
            print(abs(p.grad).mean())
            print("-next batch--------------------------------------------------------------------------------------------------------------------------")
        """
        if i%100==0:
            print('[%d/%d][%d/%d] Loss_D_img: %.4f Loss_D_vid: %.4f Loss_G_img: %.4f Loss_G_vid: %.4f'
                  % (epoch, start+opt.niter, i, len(train_loader),
                      errD_img.item(),(errD_fake_vid+errD_real_vid).item(),
                      errG_img.item(),errG_vid.item()))


        if i<3:
            frame_num_output=fake.size(0)
            half = frame_num_output//2
            num_save = 20
            if half+num_save>frame_num_output:
                num_save = 10
            vutils.save_image(real_cpu[half:half+num_save,:,:,:],
                    '%s/train/real_samples_epoch_%03d_seq%d_emo%d.png' % ("Img_folder/"+file_name,epoch,i,emo_label.item()),
                    normalize=True)
            vutils.save_image(ref_image[0,:,:,:],
                    '%s/train/reference_samples_epoch_%03d_seq%d.png' % ("Img_folder/"+file_name,epoch,i),
                    normalize=True)
            vutils.save_image(fake[half:half+num_save,:,:,:],
                    '%s/train/fake_samples_epoch_%03d_seq%d.png' % ("Img_folder/"+file_name, epoch,i),
                    normalize=True)
        i+=1



    # do checkpointing
    with torch.no_grad():
        i=0
        for sample in test_loader:
            ref_image,image_y,audio_data,emo_label = sample
            step_size = image_y.size(1)
            t_vid=np.arange(0,step_size,5).shape[0]

            image_y = image_y.to(device).view((-1,3,64,64))
            ref_image = ref_image.to(device)
            audio_data = audio_data.to(device)
            batch_size_g = image_y.size(0)

            #print("test ref image shape",ref_image.shape)
            #print("test image_y image shape",image_y.shape)
            #image_y = augment(image_y)
            #print("augment image_y image shape",image_y.shape)
            emo_label = emo_label.to(device)
            #one_hot_label = torch.zeros(1,4).to(device) #number of class
            #one_hot_label[range(1),emo_label]=1
            batch_size_d = image_y.size(0)
            label = torch.full((batch_size_d//2,), real_label, device=device)
            vid_label = torch.full((t_vid,), real_label, device=device)
            seq_label = torch.full((1,), real_label, device=device)

            #emo_labels =  torch.full((1,), emo_label.item(), device=device).long()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):

                ref_image = ref_image.to(device)
                batch_size_g = image_y.size(0)
                image_y = image_y.to(device)

                label.fill_(real_label)
                vid_label.fill_(real_label)
                seq_label.fill_(real_label)

                #random_frames = torch.LongTensor(batch_size_d//2).random_(0, batch_size_d)
                random_frames = torch.randperm(batch_size_d)[:batch_size_d//2]

                #this line is for reference concat
                output = netD_img((torch.cat((image_y[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1)))
                #output = netD_img(image_y[random_frames,:,:,:])
                errD_img_real = criterion(output, label)
                test_real_img_acc+=(output.mean().item()>0.5)
                ################################################################
                # D_vid real
                ################################################################

                output_vid = netD_vid(image_y[:,:,32:64,:],audio_data)
                errD_real_vid = criterion(output_vid, vid_label)
                test_real_vid_acc+=(output_vid.mean().item()>0.5)
                ################################################################
                # D_emo real
                ################################################################
                output,class_pred = netD_emo(image_y)
                errD_real_emo = criterion(output, seq_label) + class_loss(class_pred,emo_label)
                test_real_emo_acc+=(output.item()>0.5)
                test_real_emo_class_acc+=(torch.argmax(class_pred) == emo_label).item()
                """
                ################################################################
                # D_seq_vid real
                ################################################################
                output = netD_seq_vid(image_y)
                errD_real_seq_vid = criterion(output, seq_label)
                test_real_seq_vid_acc+=(output.item()>0.5)
                ################################################################
                """
                noise = torch.randn(step_size,128, device=device)
                fake = netG(ref_image,audio_data,noise,emo_label)

                #fake = augment(fake.squeeze(0))
                fake = fake.squeeze(0)
                label.fill_(fake_label)
                vid_label.fill_(fake_label)
                seq_label.fill_(fake_label)

                #######################################################
                #D_img loss
                #######################################################
                #this line is for reference concat
                output = netD_img((torch.cat((fake[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1)))
                #output = netD_img(fake.squeeze(0)[random_frames,:,:,:])
                errD_img_fake = criterion(output, label)
                test_fake_img_acc+=(output.mean().item()<0.5)
                ################################################################
                # D_vid
                ################################################################
                output = netD_vid(fake[:,:,32:64,:],audio_data)
                errD_fake_vid = criterion(output, vid_label)
                errD_vid_emotion_fake = errD_fake_vid
                test_fake_vid_acc+=(output.mean().item()<0.5)
                ################################################################
                # D_emo fake
                ################################################################
                output,class_pred = netD_emo(fake)
                errD_fake_emo = criterion(output, seq_label) + class_loss(class_pred,emo_label)
                test_fake_emo_acc+=(output.item()<0.5)
                """
                ################################################################
                # D_seq_vid fake
                ################################################################
                output = netD_seq_vid(fake)
                errD_fake_seq_vid = criterion(output, seq_label)
                test_fake_seq_vid_acc+=(output.item()<0.5)
                """
                #################################################################
                # G_loss
                ################################################################

                label.fill_(real_label)  # fake labels are real for generator cost
                vid_label.fill_(real_label)
                seq_label.fill_(real_label)
                #this line is for reference concat
                output = netD_img((torch.cat((fake.view(-1,3,64,64)[random_frames,:,:,:],ref_image.repeat(batch_size_d//2,1,1,1)),1)))
                #output = netD_img(fake.view(-1,3,64,64)[random_frames,:,:,:])
                errG_img = criterion(output, label)
                output = netD_vid(fake[:,:,32:64,:],audio_data)
                errG_vid =  criterion(output, vid_label)
                output,class_pred = netD_emo(fake)
                errG_emo = criterion(output, seq_label) + class_loss(class_pred,emo_label)

                #output = netD_seq_vid(fake)

                #errG_seq_vid = criterion(output, seq_label)



                errG=errG_img+errG_vid+errG_emo
                errD_vid = errD_real_vid+errD_fake_vid
                errD_img = errD_img_fake+errD_img_real
                errD_emo = errD_real_emo+errD_fake_emo
                #errD_seq_vid = errD_real_seq_vid+errD_fake_seq_vid

            #test_l1_loss += l1_loss.item()
            test_d_img_loss += errD_img.item()
            test_d_vid_loss += errD_vid.item()
            test_d_emo_loss += errD_emo.item()
            #test_d_seq_vid_loss += errD_seq_vid.item()

            #test_d_emotion_loss +=errD_vid_emotion_fake.item()

            test_g_img_loss += errG_img.item()
            test_g_vid_loss += errG_vid.item()
            test_g_emo_loss += errG_emo.item()
            #test_g_seq_vid_loss += errG_seq_vid.item()

            #test_g_emotion_loss += errG_emo.item()
            #test_gtotal_loss += errG.item()

            frame_num_output=fake.size(0)
            half = frame_num_output//2
            if i<3:
                num_save = 20
                if half+num_save>frame_num_output:
                    num_save = 10

                vutils.save_image(image_y[half:half+num_save,:,:,:],
                        '%s/test/real_samples_epoch_%03d_seq%d_emo%d.png' % ("Img_folder/"+file_name,epoch,i,emo_label.item()),
                        normalize=True)
                vutils.save_image(ref_image[0,:,:,:],
                        '%s/test/reference_samples_epoch_%03d_seq%d.png' % ("Img_folder/"+file_name,epoch,i),
                        normalize=True)
                vutils.save_image(fake[half:half+num_save,:,:,:],
                        '%s/test/fake_samples_epoch_%03d_seq%d.png' % ("Img_folder/"+file_name, epoch,i),
                        normalize=True)
            i+=1

            # statistics
            """
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(test_loader),
                     errD.item(), errG.item(),  D_G_z1, D_G_z2))
            """

    data = {'epoch': epoch,
    'train_l1_loss': train_l1_loss/len(train_loader),
    'train_d_img_loss':train_d_img_loss/len(train_loader),
    'train_d_vid_loss':train_d_vid_loss/len(train_loader),
    'train_d_emo_loss':train_d_emo_loss/len(train_loader),
    'train_g_img_loss':train_g_img_loss/len(train_loader),
    'train_g_vid_loss':train_g_vid_loss/len(train_loader),
    'train_g_emo_loss':train_g_emo_loss/len(train_loader),
    'train_gtotal_loss':train_gtotal_loss,
    'test_l1_loss':test_l1_loss/len(test_loader),
    'test_d_img_loss':test_d_img_loss/len(test_loader),
    'test_d_vid_loss':test_d_vid_loss/len(test_loader),
    'test_d_emo_loss':test_d_emo_loss/len(test_loader),
    'test_g_img_loss':test_g_img_loss/len(test_loader),
    'test_g_vid_loss':test_g_vid_loss/len(test_loader),
    'test_g_emo_loss':test_g_emo_loss/len(test_loader),
    'test_gtotal_loss':test_gtotal_loss/len(test_loader),
    'train_len':len(train_loader),
    'test_len':len(test_loader),
    'train_frames':train_frame_num,
    'test_frames':test_frame_num,
    'train_d_seq_vid_loss':train_d_seq_vid_loss/len(train_loader),
    'train_g_seq_vid_loss':train_g_seq_vid_loss/len(train_loader),
    'test_d_seq_vid_loss':test_d_seq_vid_loss/len(test_loader),
    'test_g_seq_vid_loss':test_g_seq_vid_loss/len(test_loader),

    "train_real_img_acc":test_real_img_acc/(len(train_loader)*2),
    "train_real_vid_acc":test_real_vid_acc/(len(train_loader)*2),
    "train_real_emo_acc":test_real_emo_acc/(len(train_loader)*2),
    "train_real_seq_vid_acc":test_real_seq_vid_acc/(len(train_loader)*2),
    "train_fake_img_acc":test_fake_img_acc/(len(train_loader)*2),
    "train_fake_vid_acc":test_fake_vid_acc/(len(train_loader)*2),
    "train_fake_emo_acc":test_fake_emo_acc/(len(train_loader)*2),
    "train_fake_seq_vid_acc":test_fake_seq_vid_acc/(len(train_loader)*2),

    "test_real_img_acc":test_real_img_acc/(len(test_loader)*2),
    "test_real_vid_acc":test_real_vid_acc/(len(test_loader)*2),
    "test_real_emo_acc":test_real_emo_acc/(len(test_loader)*2),
    "test_real_seq_vid_acc":test_real_seq_vid_acc/(len(test_loader)*2),
    "test_fake_img_acc":test_fake_img_acc/(len(test_loader)*2),
    "test_fake_vid_acc":test_fake_vid_acc/(len(test_loader)*2),
    "test_fake_emo_acc":test_fake_emo_acc/(len(test_loader)*2),
    "test_fake_seq_vid_acc":test_fake_seq_vid_acc/(len(test_loader)*2),

    "test_real_emo_class_acc":test_real_emo_class_acc/len(test_loader),
    "train_real_emo_class_acc":train_real_emo_class_acc/len(train_loader)
    }
    if opt.log == "True":
        my_file=open(plot_file, "a")
    df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
    df.to_csv(my_file, header=False,index=False)
    my_file.close()

    torch.save(netG.state_dict(), '%s/Models/%s/netG_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netD_img.state_dict(), '%s/Models/%s/netD_img_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netD_vid.state_dict(), '%s/Models/%s/netD_vid_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netD_emo.state_dict(), '%s/Models/%s/netD_emo_epoch_%d.pth' % (opt.outf, file_name,epoch))
    #torch.save(netD_seq_vid.state_dict(), '%s/Models/%s/netD_seq_vid_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netG.state_dict(), '%s/Models/%s/netG_epoch_%d.pth' % (opt.outf, file_name,epoch))
    ##########################################################################################################
    #Save optimizers
    ##########################################################################################################
    torch.save(optimizerD_img.state_dict(), '%s/Models/%s/optimizerD_img.pth' % (opt.outf, file_name))
    torch.save(optimizerD_vid.state_dict(), '%s/Models/%s/optimizerD_vid.pth' % (opt.outf, file_name))
    torch.save(optimizerD_emo.state_dict(), '%s/Models/%s/optimizerD_emo.pth' % (opt.outf, file_name))
    #torch.save(optimizerD_seq_vid.state_dict(), '%s/Models/%s/optimizerD_seq_vid.pth' % (opt.outf, file_name))
    torch.save(optimizerG.state_dict(), '%s/Models/%s/optimizerG.pth' % (opt.outf, file_name))
