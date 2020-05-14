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
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

parser.add_argument('--log', default='False')

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


        random.shuffle(self.keys)

        self.keys = self.keys[0:1000]

        video_folder_list = os.listdir(video_root_dir)
        self.video_list = []
        """
        for i in tqdm(video_folder_list):
            self.video_list = self.video_list  + [i+"/"+x for x in os.listdir(os.path.join(video_root_dir,i))]
        """
        self.img_dict = {}
        for i in tqdm(self.keys):
            self.img_dict[i] = self.read_images(os.path.join(self.video_root_dir,i),self.audio_dict[i].shape[0])
            self.frame_num+=self.img_dict[i].size(0)


    def __len__(self):
        return len(self.keys)


    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py
    def get_frame_num(self):
        return self.frame_num

    def read_images(self, video_dir,frame_num):
        image_list=os.listdir(video_dir)

        X = []

        for i in range(1,frame_num+1):
            image = Image.open(os.path.join(video_dir,'image-{:d}.jpeg'.format(i)))

            if self.transform:
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
        """
        image_y = self.read_images(sequence_name,frame_num)
        """
        image_y = self.img_dict[selected_elem]

        ref_image = Image.open(os.path.join(sequence_name,'image-{:d}.jpeg'.format(np.random.randint(frame_num)+1)))
        ref_image = self.transform(ref_image)

        return ref_image,image_y,audio_data,label

dataset_train = Audio_video_dataset(h5_file="../data/audio/4class_indep/audio_features_4class(1pad)_train.hdf5",
                                           video_root_dir='../data/video/cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
dataset_test = Audio_video_dataset(h5_file="../data/audio/4class_indep/audio_features_4class(1pad)_test.hdf5",
                                           video_root_dir='../data/video/cropped_face_frames',
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


nc=3


device = torch.device("cuda:0")# if opt.cuda else "cpu")
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

        ngf = 64
        ndf = 64
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.img_enc_l5 = nn.Conv2d(ndf * 2, self.img_latent_dim, 4, 1, 0, bias=False)

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, ndf//2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//2, ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf* 2, ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 1, 1, 0, bias=False),
        )

        self.audio_linear = nn.Linear(ndf*60*3,self.aud_latent_dim)

        self.lstm_hiddend_dim = self.aud_latent_dim*2

        self.encoded_dim = self.img_latent_dim + self.lstm_hiddend_dim + 100

        self.decoder_l1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.encoded_dim , ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.decoder_l2 = nn.Sequential(    # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.decoder_l3 = nn.Sequential(    # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.decoder_l4 = nn.Sequential(    # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.decoder_l5 = nn.Sequential(    # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf*2,      3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.label_emb = nn.Embedding(100, 100)

        self.lstm = nn.LSTM(self.aud_latent_dim, self.lstm_hiddend_dim, 1, batch_first=True)
        self.noise_lstm = nn.LSTM(150,100,1, batch_first=True)


    def forward(self,ref_image,audio_data,noise,emo_label):

        batch_size = ref_image.size(0)
        step_size = audio_data.size(1)

        enc_l1 = self.img_enc_l1(ref_image)
        #print(enc_l1.shape)
        enc_l2 = self.img_enc_l2(enc_l1)
        #print(enc_l2.shape)
        enc_l3 = self.img_enc_l3(enc_l2)
        #print(enc_l3.shape)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print(enc_l4.shape)
        enc_l5 = self.img_enc_l5(enc_l4)
        #print(enc_l5.shape)
        #print("ref",x1.shape)
        #print("audio befoer",audio_data.shape)
        x2 = self.audio_encoder(audio_data.view((-1,1,60,3)))
        #print("audio",x2.shape)
        x2 = self.audio_linear(x2.view(-1,64*60*3))
        #print("audio linear",x2.shape)
        #print("one hot label",one_hot_label.shape)

        h0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)

        h1 = torch.zeros(1, batch_size, 100).to(device)
        c1 = torch.zeros(1, batch_size, 100).to(device)

        x, _ = self.lstm(x2.view((batch_size,-1,self.aud_latent_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        embed = self.label_emb(emo_label).repeat(step_size,1,1,1)
        #print("embed shape",embed.shape)
        #print("noise shape",noise.shape)
        y = torch.cat((noise.view(-1,1,1,50),embed),axis=3)
        #print("y shape",y.shape)
        y, _ = self.noise_lstm(y.view((batch_size,-1,150)), (h1, c1))
        #print("y cat shape",y.shape)

        z_latent = torch.cat((enc_l5.repeat(step_size,1,1,1),x.view(-1,self.lstm_hiddend_dim,1,1),y.view(-1,100,1,1)),axis=1)

        dec_l1 = self.decoder_l1(z_latent.reshape((-1,self.encoded_dim,1,1)))

        dec_l2 = self.decoder_l2(torch.cat((enc_l4.repeat(step_size,1,1,1),dec_l1),1))
        #print(dec_l2.shape)
        dec_l3 = self.decoder_l3(dec_l2)
        #print(dec_l3.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l4 = self.decoder_l4(torch.cat((enc_l2.repeat(step_size,1,1,1),dec_l3),1))
        #print(dec_l4.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l5 = self.decoder_l5(torch.cat((enc_l1.repeat(step_size,1,1,1),dec_l4),1))

        return dec_l5.view((batch_size,-1,3,64,64))


netG = Generator(128,256,100).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

#netG.load_state_dict(torch.load("netG_epoch_149_continue2.pth"))

#torch.save(netG.main.state_dict(), 'netG_epoch_149_continue2.pt')

class Discriminator_img(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_img, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
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
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD_img)




class Discriminator_vid(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator_vid, self).__init__()
        self.latent_dim = latent_dim
        ngf = 32
        ndf = 32
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, ndf//2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//2, ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf* 2, ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 1, 1, 0, bias=False),
        )

        self.img_linear = nn.Linear(ndf * 2 * 4* 2,self.latent_dim)

        self.audio_linear = nn.Linear(ndf*60*3,self.latent_dim)

        self.encoded_dim = self.latent_dim*2

        self.lstm_hiddend_dim = self.encoded_dim*2

        self.lstm = nn.LSTM(self.encoded_dim, self.lstm_hiddend_dim, 1, batch_first=True)


        self.real_vid = nn.Sequential(nn.Linear(self.lstm_hiddend_dim,1),nn.Sigmoid())

    def forward(self,imgs,audio_data):


        step_size = audio_data.size(1)

        enc_l1 = self.img_enc_l1(imgs.view((-1,3,32,64)))
        enc_l2 = self.img_enc_l2(enc_l1)
        enc_l3 = self.img_enc_l3(enc_l2)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print(enc_l4.shape)
        x1 = self.img_linear(enc_l4.view((-1,ndf * 2 * 4* 2)))
        #print(x1.shape)
        x2 = self.audio_encoder(audio_data.view((-1,1,60,3)))

        x2 = self.audio_linear(x2.view(-1,32*60*3))

        z_latent = torch.cat((x1.unsqueeze(2).unsqueeze(3),x2.view(-1,self.latent_dim,1,1)),axis=1)


        h0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(z_latent.view((1,-1,self.encoded_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        real_vid = self.real_vid(x[:,np.arange(0,step_size,5),:])

        return real_vid.squeeze(0).squeeze(1)




netD_vid = Discriminator_vid(256).to(device)
netD_vid.apply(weights_init)
print(netD_vid)

class Discriminator_emotion(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator_emotion, self).__init__()
        self.latent_dim = latent_dim
        ngf = 32
        ndf = 32
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l2 = nn.Sequential(    # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l3 = nn.Sequential(    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.img_enc_l4 = nn.Sequential(    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.img_linear = nn.Linear(ndf * 2 * 4* 2,self.latent_dim)

        self.lstm_hiddend_dim = self.latent_dim*2

        self.lstm = nn.LSTM(self.latent_dim, self.lstm_hiddend_dim, 1, batch_first=True)

        self.lstm_linear= nn.Sequential(nn.LeakyReLU(0.2, inplace=False),nn.Linear(self.lstm_hiddend_dim,200))

        self.label_emb = nn.Sequential(nn.Embedding(100, 100),nn.Linear(100,100))

        self.classify =  nn.Sequential(nn.LeakyReLU(0.2, inplace=False),nn.Linear(300,1),nn.Sigmoid())
    def forward(self,imgs,emo_label):


        enc_l1 = self.img_enc_l1(imgs.view((-1,3,32,64)))
        enc_l2 = self.img_enc_l2(enc_l1)
        enc_l3 = self.img_enc_l3(enc_l2)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print(enc_l4.shape)
        x1 = self.img_linear(enc_l4.view((-1,ndf * 2 * 4* 2)))
        #print(x1.shape)

        h0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(x1.view((1,-1,self.latent_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        emo = self.lstm_linear(x[:,-1,:])
        #print("emo shape",emo.shape)
        label_emb = self.label_emb(emo_label)
        #print("label_emb shape",label_emb.shape)
        out = self.classify(torch.cat((emo,label_emb),axis=1))

        return out.squeeze(0)

netD_emo = Discriminator_emotion(256).to(device)
netD_emo.apply(weights_init)
print(netD_emo)



criterion = nn.BCELoss()
L1_loss = nn.L1Loss()


real_label = 1
fake_label = 0

# setup optimizer
optimizerD_img = optim.Adam(netD_img.parameters(), lr=0.0003, betas=(opt.beta1, 0.999))
optimizerD_vid = optim.Adam(netD_vid.parameters(), lr=0.0003, betas=(opt.beta1, 0.999))
optimizerD_emo = optim.Adam(netD_emo.parameters(), lr=0.003, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

train_frame_num = dataset_train.get_frame_num()
test_frame_num =  dataset_test.get_frame_num()


print("Train frame num: ",train_frame_num)
print("Test frame num: ",test_frame_num)

now=datetime.now()
plot_file="lip_gan_v4_binary"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


for epoch in range(opt.niter):

    train_l1_loss = 0
    train_d_img_loss = 0
    train_d_vid_loss = 0
    train_d_emo_loss = 0
    train_d_emotion_loss = 0

    train_g_img_loss = 0
    train_g_vid_loss = 0
    train_g_emo_loss = 0
    train_g_emotion_loss = 0
    train_gtotal_loss = 0
    train_dx = 0

    test_l1_loss = 0
    test_d_img_loss = 0
    test_d_vid_loss = 0
    test_d_emo_loss = 0
    test_d_emotion_loss = 0

    test_g_img_loss = 0
    test_g_vid_loss = 0
    test_g_emo_loss = 0
    test_g_emotion_loss = 0
    test_gtotal_loss = 0


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
        batch_size_d = real_cpu.size(0)
        step_size = image_y.size(1)
        emo_label = emo_label.to(device)
        label = torch.full((batch_size_d,), real_label, device=device)
        #emo_labels =  torch.full((1,), emo_label.item(), device=device).long()

        #one_hot_label = torch.zeros(1,4).to(device) #number of class
        #one_hot_label[:,emo_label]=1

        t_vid=np.arange(0,step_size,5).shape[0]

        vid_label = torch.full((t_vid,), real_label, device=device)
        seq_label = torch.full((1,), real_label, device=device)
        #label[np.random.randint(batch_size,size=(batch_size//20))]=fake_label

        output = netD_img(real_cpu)
        errD_img_real = criterion(output, label)
        errD_img_real.backward()
        D_x = output.mean().item()

        ################################################################
        # D_vid real
        ################################################################
        netD_vid.zero_grad()

        output_vid = netD_vid(real_cpu[:,:,32:64,:],audio_data)
        errD_real_vid = criterion(output_vid, vid_label)
        errD_real_vid.backward()
        #D_x = output.mean().item()

        ################################################################
        # D_emo real
        ################################################################
        netD_emo.zero_grad()
        output_seq = netD_emo(real_cpu,emo_label)
        #print("seq label",seq_label.shape)
        errD_real_emo = criterion(output_seq, seq_label)
        errD_real_emo.backward()

        ################################################################
        # D_img fake
        ################################################################
        # train with fake
        ref_image = ref_image.to(device)
        batch_size_g = image_y.size(0)

        #one_hot_label = torch.zeros(batch_size_g,4).to(device) #number of class
        #one_hot_label[range(batch_size_g),emo_label]=1


        noise = torch.randn(step_size,50, device=device)
        fake = netG(ref_image,audio_data,noise,emo_label)
        label.fill_(fake_label)
        vid_label.fill_(fake_label)
        seq_label.fill_(fake_label)
        #l1_loss = L1_loss(fake.squeeze(0)[:,:,32:64,:],real_cpu[:,:,32:64,:])*5

        output = netD_img(fake.view(-1,3,64,64).detach())
        errD_img_fake = criterion(output, label)
        errD_img_fake.backward()

        D_G_z1 = output.mean().item()
        errD_img = errD_img_real + errD_img_fake
        optimizerD_img.step()


        ################################################################
        # D_vid fake
        ################################################################

        output = netD_vid(fake.detach().squeeze(0)[:,:,32:64,:],audio_data)
        errD_fake_vid = criterion(output, vid_label)
        errD_fake_vid.backward()
        optimizerD_vid.step()

        ################################################################
        # D_emo fake
        ################################################################
        output_seq = netD_emo(fake.detach().squeeze(0),emo_label)
        errD_fake_emo = criterion(output_seq, seq_label)
        errD_fake_emo.backward()
        optimizerD_emo.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        vid_label.fill_(real_label)
        seq_label.fill_(real_label)
        output = netD_img(fake.view(-1,3,64,64))
        errG_img = criterion(output, label)

        output = netD_vid(fake.squeeze(0)[:,:,32:64,:],audio_data)
        errG_vid =  criterion(output, vid_label)

        output = netD_emo(fake.squeeze(0),emo_label)
        errG_emo =  criterion(output, seq_label)

        errG= errG_img +errG_vid + errG_emo

        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()



        """
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        """

        train_d_img_loss += errD_img.item()
        train_d_vid_loss += (errD_fake_vid+errD_real_vid).item()
        train_d_emo_loss += (errD_fake_emo+errD_real_emo).item()
        #train_d_emotion_loss += (errD_fake_emotion_vid+errD_real_emotion_vid).item()

        #train_l1_loss += l1_loss.item()
        train_g_img_loss +=errG_img.item()
        train_g_vid_loss += errG_vid.item()
        #train_g_emotion_loss += errG_emo.item()
        train_gtotal_loss += errG.item()
        train_g_emo_loss +=errG_emo.item()
        """
        for p in netD_emo.parameters(): #,model._all_weights[0]): #prints gradients below
            #if n[:6] == 'weight':

            print('===========\ngradient:{}\n----------\n{}----------\n{}'.format(p.grad,p.grad.shape,p.grad.mean()))
        """
        if i%100==0:
            print('[%d/%d][%d/%d] Loss_D_img: %.4f Loss_D_vid: %.4f Loss_G_img: %.4f Loss_G_vid: %.4f'
                  % (epoch, opt.niter, i, len(train_loader),
                      errD_img.item(),(errD_fake_vid+errD_real_vid).item(),
                      errG_img.item(),errG_vid.item()))


        if i<3:
            frame_num_output=fake.size(1)
            half = frame_num_output//2
            num_save = 20
            if half+num_save>frame_num_output:
                num_save = 10
            vutils.save_image(image_y[0,half:half+num_save,:,:,:],
                    '%s/train/real_samples_epoch_%03d_seq%d.png' % ("Img_folder",epoch,i),
                    normalize=True)
            vutils.save_image(ref_image[0,:,:,:],
                    '%s/train/reference_samples_epoch_%03d_seq%d.png' % ("Img_folder",epoch,i),
                    normalize=True)
            vutils.save_image(fake[0,half:half+num_save,:,:,:],
                    '%s/train/fake_samples_epoch_%03d_seq%d.png' % ("Img_folder", epoch,i),
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

            emo_label = emo_label.to(device)
            #one_hot_label = torch.zeros(1,4).to(device) #number of class
            #one_hot_label[range(1),emo_label]=1
            batch_size_d = image_y.size(0)
            label = torch.full((batch_size_d,), real_label, device=device)
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

                output = netD_img(image_y)
                errD_img_real = criterion(output, label)

                ################################################################
                # D_vid real
                ################################################################

                output_vid = netD_vid(image_y[:,:,32:64,:],audio_data)
                errD_real_vid = criterion(output_vid, vid_label)

                ################################################################
                # D_emo real
                ################################################################
                output = netD_emo(image_y,emo_label)
                errD_real_emo = criterion(output, seq_label)
                ################################################################
                noise = torch.randn(step_size,50, device=device)
                fake = netG(ref_image,audio_data,noise,emo_label)
                label.fill_(fake_label)
                vid_label.fill_(fake_label)
                seq_label.fill_(fake_label)
                #######################################################
                #L1 loss
                #######################################################

                #l1_loss = L1_loss(fake.squeeze(0)[:,:,32:64,:],image_y.squeeze(0)[:,:,32:64,:])*5
                #######################################################
                #D_img loss
                #######################################################
                output = netD_img(fake.squeeze(0))
                errD_img_fake = criterion(output, label)
                ################################################################
                # D_vid
                ################################################################
                output = netD_vid(fake.squeeze(0)[:,:,32:64,:],audio_data)
                errD_fake_vid = criterion(output, vid_label)
                errD_vid_emotion_fake = errD_fake_vid
                ################################################################
                # D_emo fake
                ################################################################
                output = netD_emo(fake.squeeze(0),emo_label)
                errD_fake_emo = criterion(output, seq_label)
                #################################################################
                # G_loss
                ################################################################

                label.fill_(real_label)  # fake labels are real for generator cost
                vid_label.fill_(real_label)
                seq_label.fill_(real_label)
                output = netD_img(fake.view(-1,3,64,64))
                errG_img = criterion(output, label)
                output = netD_vid(fake.squeeze(0)[:,:,32:64,:],audio_data)
                errG_vid =  criterion(output, vid_label)
                output = netD_emo(fake.squeeze(0),emo_label)
                errG_emo = criterion(output, seq_label)


                errG=errG_img+errG_vid+errG_emo
                errD_vid = errD_real_vid+errD_fake_vid
                errD_img = errD_img_fake+errD_img_real
                errD_emo = errD_real_emo+errD_fake_emo

            #test_l1_loss += l1_loss.item()
            test_d_img_loss += errD_img.item()
            test_d_vid_loss += errD_vid.item()
            test_d_emo_loss += errD_emo.item()

            #test_d_emotion_loss +=errD_vid_emotion_fake.item()

            test_g_img_loss += errG_img.item()
            test_g_vid_loss += errG_vid.item()
            test_g_emo_loss += errG_emo.item()
            #test_g_emotion_loss += errG_emo.item()
            #test_gtotal_loss += errG.item()

            frame_num_output=fake.size(1)
            half = frame_num_output//2
            if i<3:
                num_save = 20
                if half+num_save>frame_num_output:
                    num_save = 10

                vutils.save_image(image_y[half:half+num_save,:,:,:],
                        '%s/test/real_samples_epoch_%03d_seq%d.png' % ("Img_folder",epoch,i),
                        normalize=True)
                vutils.save_image(ref_image[0,:,:,:],
                        '%s/test/reference_samples_epoch_%03d_seq%d.png' % ("Img_folder",epoch,i),
                        normalize=True)
                vutils.save_image(fake[0,half:half+num_save,:,:,:],
                        '%s/test/fake_samples_epoch_%03d_seq%d.png' % ("Img_folder", epoch,i),
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
    'test_frames':test_frame_num
    }
    if opt.log == "True":
        my_file=open(plot_file, "a")
    df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
    df.to_csv(my_file, header=False,index=False)
    my_file.close()
    if epoch%2==0:
        torch.save(netG.state_dict(), '%s/Models/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD_img.state_dict(), '%s/Models/netD_epoch_%d.pth' % (opt.outf, epoch))
