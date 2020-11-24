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
parser.add_argument('--small_test',default='False')

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


class Landmark_video_dataset(Dataset):


    def __init__(self, video_root_dir,train,transform_img=None,transform_land=None):


        all_videos = os.listdir(os.path.join(video_root_dir,"imgs"))
        all_videos.sort()

        self.video_root_dir = video_root_dir
        self.transform_img = transform_img
        self.transform_land = transform_land

        self.train_sub =np.array([1001, 1002, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
        1013, 1014, 1016, 1017, 1018, 1022, 1025, 1026, 1027, 1029, 1031,
        1032, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
        1044, 1045, 1046, 1047, 1048, 1049, 1051, 1053, 1054, 1055, 1057,
        1059, 1060, 1061, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070,
        1071, 1072, 1075, 1076, 1077, 1078, 1079, 1080, 1083, 1084, 1085,
        1086, 1087, 1088, 1090, 1091]).astype(str)

        self.val_sub = np.array([1003, 1019, 1023, 1024, 1028, 1050, 1056, 1058, 1073, 1074]).astype(str)

        if train:
            search_list = self.train_sub
        else:
            search_list = self.val_sub

        self.file_list = []

        for i in all_videos:
            for j in search_list:
                if j in i:
                    self.file_list.append(i)
                    break
        if opt.small_test=='True' or opt.small_test=='true':
            self.file_list = self.file_list[0:10]
        try:
            self.file_list.remove("1064_IEO_DIS_MD")#more than 1000 frames (faulty)
            self.file_list.remove("1064_TIE_SAD_XX")#18 frames
            self.file_list.remove("1076_MTI_NEU_XX")#onyl 3 frames
        except:
            pass
        """
        train
        [1001, 1002, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
        1013, 1014, 1016, 1017, 1018, 1022, 1025, 1026, 1027, 1029, 1031,
        1032, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
        1044, 1045, 1046, 1047, 1048, 1049, 1051, 1053, 1054, 1055, 1057,
        1059, 1060, 1061, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070,
        1071, 1072, 1075, 1076, 1077, 1078, 1079, 1080, 1083, 1084, 1085,
        1086, 1087, 1088, 1090, 1091])
        """
        # validation# [1003, 1019, 1023, 1024, 1028, 1050, 1056, 1058, 1073, 1074]
        # test # 1015, 1020, 1021, 1030, 1033, 1052, 1062, 1081, 1082, 1089

    def __len__(self):
        return len(self.file_list)

    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py

    def read_images(self, video_dir,flip,img_transform):

        image_list=os.listdir(video_dir)
        frame_num = len(image_list)

        X = []
        brightness = torch.rand(1)*0.9+0.4
        contrast = torch.rand(1)+0.5
        saturation = torch.rand(1)*1.5
        #hue = torch.rand(1)-0.5/5


        for i in range(0,frame_num):
            image = Image.open(os.path.join(video_dir,'image-{:d}.jpg'.format(i)))
            #image = image.convert('RGB')

            if img_transform==True:

                if flip==True:
                    image = hflip(image)
                """
                image = adjust_brightness(image,brightness_factor=brightness) #0-2
                image = adjust_contrast(image,contrast_factor=contrast) #0-2
                image = adjust_saturation(image,saturation_factor=saturation)
                #image  = adjust_hue(image,hue_factor = hue )#-0.5,0.5
                """
                image = self.transform_img(image)
            else:
                if flip==True:
                    image = hflip(image)
                """
                image = adjust_brightness(image,brightness_factor=brightness) #0-2
                image = adjust_contrast(image,contrast_factor=contrast) #0-2
                image = adjust_saturation(image,saturation_factor=saturation)
                #image  = adjust_hue(image,hue_factor = hue )#-0.5,0.5
                """
                """
                image = np.array(image)
                image = np.expand_dims(image, axis=2)
                print(image.shape)
                image = Image.fromarray(image)
                """
                image = self.transform_land(image)


            X.append(image)
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.file_list[idx]

        sequence_imgs = os.path.join(self.video_root_dir,"imgs",selected_elem)
        sequence_landmarks = os.path.join(self.video_root_dir,"landmarks",selected_elem)

        flip = torch.rand(1).item()>0.5

        image_y = self.read_images(sequence_imgs,flip,img_transform=True)

        landmarks = self.read_images(sequence_landmarks,flip,img_transform=False)

        ref_image = image_y[0,:,:,:]
        ref_landmark = landmarks[0,:,:,:]

        return ref_image,image_y,ref_landmark,landmarks

dataset_train = Landmark_video_dataset(video_root_dir='../data/video/cropped_face_frames_land',
                                           train=True,
                                           transform_img=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
                                         ,transform_land=transforms.Compose([
                                            transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((64,64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, ), (0.5, ))
                                         ])
                                           )
dataset_val = Landmark_video_dataset(video_root_dir='../data/video/cropped_face_frames_land',
                                           train=False,
                                           transform_img=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                           ])
                                         ,transform_land=transforms.Compose([
                                            transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((64,64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, ), (0.5, ))
                                         ])
                                           )



batch_size = 1


def customBatchBuilder(samples):
    ref_image,image_y,ref_landmark,landmarks = zip(*samples)

    landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0)
    image_y = pad_sequence(image_y, batch_first=True, padding_value=0)

    ref_img_tensor = torch.Tensor(ref_image[0]).unsqueeze(0)
    ref_landmark_tensor = torch.Tensor(ref_landmark[0]).unsqueeze(0)
    for i in range(1,len(ref_image)):
        ref_img_tensor=torch.cat((ref_img_tensor,torch.Tensor(ref_image[i]).unsqueeze(0)),axis=0)
        ref_landmark_tensor=torch.cat((ref_landmark_tensor,torch.Tensor(ref_landmark[i]).unsqueeze(0)),axis=0)

    return ref_img_tensor,image_y,ref_landmark_tensor,landmarks



train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)




val_loader = DataLoader(dataset_val, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)



train_set_size = len(dataset_train)
val_set_size = len(dataset_val)

print("Train set size:",train_set_size)
print("Val set size:",val_set_size)

print("train_loader",len(train_loader))
print("val_loader",len(val_loader))


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
    def __init__(self,img_latent_dim):
        super(Generator, self).__init__()
        self.img_latent_dim = img_latent_dim

        self.ndf = 64
        self.ngf = 64
        self.img_enc_l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(5, self.ndf, 4, 2, 1, bias=False),
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

        self.lstm_hiddend_dim = self.ndf*8

        self.lstm = nn.LSTM(self.lstm_hiddend_dim, self.lstm_hiddend_dim, 1, batch_first=True)

        self.encoded_dim =  self.ndf*8*2


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

    def forward(self,ref_image,ref_landmark,landmarks):

        batch_size = ref_image.size(0)
        step_size = landmarks.size(0)

        #print("landmarks size",landmarks.shape)
        #print("ref_landmark size",ref_landmark.shape)
        #print("ref_image size",ref_image.shape)

        imput = torch.cat((ref_image.repeat(step_size,1,1,1),ref_landmark.repeat(step_size,1,1,1),landmarks),axis=1)


        enc_l1 = self.img_enc_l1(imput)
        #print("enc1",enc_l1.shape)
        enc_l2 = self.img_enc_l2(enc_l1)
        #print("enc2",enc_l2.shape)
        enc_l3 = self.img_enc_l3(enc_l2)
        #print("enc3",enc_l3.shape)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print("enc4",enc_l4.shape)
        enc_l5 = self.img_enc_l5(enc_l4)
        #print("enc5",enc_l5.shape)

        h0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(enc_l5.view(batch_size,-1,self.lstm_hiddend_dim), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        #print("x shape",x.shape)

        z_latent = torch.cat((enc_l5.view(batch_size,-1,self.lstm_hiddend_dim),x.view(batch_size,-1,self.lstm_hiddend_dim)),axis=2)
        #print("z_latent shape",z_latent.shape)

        dec_l1 = self.decoder_l1(z_latent.reshape((-1,self.encoded_dim,1,1)))
        #print("dec1",dec_l1.shape)
        dec_l2 = self.decoder_l2(torch.cat((enc_l4,dec_l1),1))
        #print("dec2",dec_l2.shape)
        dec_l3 = self.decoder_l3(torch.cat((enc_l3,dec_l2),1))
        #print("dec3",dec_l3.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l4 = self.decoder_l4(torch.cat((enc_l2,dec_l3),1))
        #print("dec4",dec_l4.shape)
        #print(enc_l2.shape,enc_l2.repeat(step_size,1,1,1).shape,dec_l3.shape)
        dec_l5 = self.decoder_l5(torch.cat((enc_l1,dec_l4),1))
        #print("dec_l5",dec_l5.shape)
        return dec_l5.view((batch_size,-1,3,64,64))

netG = Generator(128).to(device)
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
            nn.Conv2d(8, self.ndf, 4, 2, 1, bias=False),
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
        self.img_enc_l5 = nn.Sequential(    # state size. (ndf*4) x 4 x 4
            nn.Conv2d(self.ndf * 2, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d( self.ndf*2,affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.img_linear = nn.Linear(self.ndf * 2 * 2* 2,self.latent_dim)

        self.lstm_hiddend_dim = self.latent_dim*4

        self.lstm = nn.LSTM(self.latent_dim, self.lstm_hiddend_dim, 1, batch_first=True)

        self.real_vid = nn.Sequential(nn.LeakyReLU(0.2),nn.Linear(self.lstm_hiddend_dim,1),nn.Sigmoid())

    def forward(self,imgs):
        #print("imgs shape",imgs.shape)
        enc_l1 = self.img_enc_l1(imgs.view((-1,3,64,64)))
        enc_l2 = self.img_enc_l2(enc_l1)
        enc_l3 = self.img_enc_l3(enc_l2)
        enc_l4 = self.img_enc_l4(enc_l3)
        #print("encl4",enc_l4.shape)
        enc_l5 = self.img_enc_l5(enc_l4)
        #print("encl5",enc_l5.shape)
        x1 = self.img_linear(enc_l5.view((-1,self.ndf * 2 * 2* 2)))
        #print(x1.shape)

        h0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)
        c0 = torch.zeros(1, 1, self.lstm_hiddend_dim).to(device)

        x, _ = self.lstm(x1.view((1,-1,self.latent_dim)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print("x size",x.shape)
        real_vid = self.real_vid(x[:,-1,:])
        #print("real_vid shape",real_vid.shape)
        return real_vid.squeeze(0)




netD_vid = Discriminator_vid(128).to(device)
netD_vid.apply(weights_init)
print(netD_vid)


criterion = nn.BCELoss()
L1_loss = nn.L1Loss()
class_loss = nn.CrossEntropyLoss()

real_label = 1
fake_label = 0
#https://discuss.pytorch.org/t/loading-optimizer-dict-starts-training-from-initial-lr/36328/2
file_name = "landmark_ganv1_l1"
img_folder_dir = os.path.join("Landmark_Img_folder",file_name)
if not os.path.exists(img_folder_dir):
    os.makedirs(img_folder_dir)
    os.makedirs(os.path.join(img_folder_dir,"val"))
    os.makedirs(os.path.join(img_folder_dir,"train"))

model_dir = os.path.join("Landmark_Models",file_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_dir = os.listdir(os.path.join("Landmark_Models",file_name))




now=datetime.now()

plot_file="./Landmark_Img_folder/"+file_name+"/"+file_name+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"

# setup optimizer
optimizerD_img = optim.Adam(netD_img.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
optimizerD_vid = optim.Adam(netD_vid.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
#optimizerD_seq_vid = optim.Adam(netD_seq_vid.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

start = 0

if (len(model_dir)!=0 or opt.load_epoch != '') and opt.start_over=='False':
    print("Loading old save")
    list_models = [int(i.split(".")[0].split("epoch_")[1]) for i in model_dir if "netD_img" in i]
    list_models.sort()
    print(list_models)
    load_epoch = list_models[-1]
    print(load_epoch)
    load_epoch=load_epoch

    netD_img.load_state_dict(torch.load('%s/Landmark_Models/%s/netD_img_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    netD_vid.load_state_dict(torch.load('%s/Landmark_Models/%s/netD_vid_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))
    netG.load_state_dict(torch.load('%s/Landmark_Models/%s/netG_epoch_%s.pth' % (opt.outf, file_name,load_epoch)))

    ##########################################################################################################################
    optimizerD_img.load_state_dict(torch.load('%s/Landmark_Models/%s/optimizerD_img.pth' % (opt.outf, file_name)))
    optimizerD_vid.load_state_dict(torch.load('%s/Landmark_Models/%s/optimizerD_vid.pth' % (opt.outf, file_name)))
    optimizerG.load_state_dict(torch.load('%s/Landmark_Models/%s/optimizerG.pth' % (opt.outf, file_name)))

    start = int(load_epoch)+1

    #if opt.load_plot_file != '':
    plot_file_dir_list  = os.listdir(os.path.join("Landmark_Img_folder",file_name))
    print(plot_file_dir_list)
    list_plots = [i for i in plot_file_dir_list if "csv" in i]
    print(list_plots)
    plot_file="./Landmark_Img_folder/"+file_name+"/"+list_plots[0]


"""
train_frame_num = dataset_train.get_frame_num()
test_frame_num =  dataset_test.get_frame_num()
"""
if opt.schedular == "True":
    schedulerD_img = StepLR(optimizerD_img, step_size=15, gamma=0.1)
    schedulerD_vid = StepLR(optimizerD_vid, step_size=15, gamma=0.1)
    schedulerG = StepLR(optimizerG, step_size=15, gamma=0.1)




"""
print("Train frame num: ",train_frame_num)
print("Test frame num: ",test_frame_num)
"""


if not os.path.exists("./Landmark_Img_folder/"+file_name):
    os.mkdir("./Landmark_Img_folder/"+file_name)

if not os.path.exists("./Landmark_Models/"+file_name):
    os.mkdir("./Landmark_Models/"+file_name)




for epoch in range(start,start+opt.niter):

    train_d_img_loss = 0
    train_d_vid_loss = 0

    train_real_img_acc= 0
    train_real_vid_acc = 0

    train_fake_img_acc= 0
    train_fake_vid_acc = 0

    train_g_img_loss = 0
    train_g_vid_loss = 0

    train_gtotal_loss = 0
    train_dx = 0

    val_d_img_loss = 0
    val_d_vid_loss = 0

    val_g_img_loss = 0
    val_g_vid_loss = 0
    val_gtotal_loss = 0

    val_real_img_acc = 0
    val_real_vid_acc = 0

    val_fake_img_acc = 0
    val_fake_vid_acc = 0

    train_g_l1_loss = 0
    val_g_l1_loss = 0


    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        ref_image,image_y,ref_landmark,landmarks = data


        image_y = image_y.to(device).view((-1,3,64,64))
        step_size = image_y.size(0)
        ref_image = ref_image.to(device)
        ref_landmark = ref_landmark.to(device)
        landmarks = landmarks.to(device).view((-1,1,64,64))


        label = torch.full((step_size//2,), real_label, device=device)
        vid_label = torch.full((1,), real_label, device=device)

        ################################################################
        # D_img fake
        ################################################################
        netD_img.zero_grad()

        random_frames = torch.randperm(step_size)[:step_size//2]
        #print("landmarks shape",landmarks.shape)
        output = netD_img(torch.cat((image_y[random_frames,:,:,:],ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
        errD_img_real = criterion(output, label)
        errD_img_real.backward()
        train_real_img_acc+=(output.mean().item()>0.5)
        ################################################################
        # D_vid real
        ################################################################
        netD_vid.zero_grad()

        output_vid = netD_vid(image_y)
        errD_real_vid = criterion(output_vid, vid_label)
        errD_real_vid.backward()
        train_real_vid_acc+=(output_vid.mean().item()>0.5)

        ################################################################
        # D_img fake
        ################################################################
        # train with fake

        batch_size_g = image_y.size(0)

        #noise = torch.randn(step_size,128, device=device)
        #print("Netg start")
        fake = netG(ref_image,ref_landmark,landmarks)

        #fake = augment(fake.squeeze(0))
        fake = fake.squeeze(0)

        label.fill_(fake_label)
        vid_label.fill_(fake_label)

        #the line below is for reference concat
        output = netD_img(torch.cat((fake[random_frames,:,:,:].detach(),ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
        errD_img_fake = criterion(output, label)
        errD_img_fake.backward()
        train_fake_img_acc+=(output.mean().item()<0.5)

        errD_img = errD_img_real + errD_img_fake

        if opt.schedular == "True":
            schedulerD_img.step()
        else:
            optimizerD_img.step()

        #print("netD_img end")
        ################################################################
        # D_vid fake
        ################################################################

        output = netD_vid(fake.detach())
        errD_fake_vid = criterion(output, vid_label)
        errD_fake_vid.backward()
        train_fake_vid_acc+=(output.mean().item()<0.5)
        if opt.schedular == "True":
            schedulerD_vid.step()
        else:
            optimizerD_vid.step()
        #print("netD_vid end")
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        #print("Update netg start")
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        vid_label.fill_(real_label)
        #the line below is for reference concat
        output = netD_img(torch.cat((fake[random_frames,:,:,:],ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
        errG_img = criterion(output, label)

        output = netD_vid(fake)
        errG_vid =  criterion(output, vid_label)

        errG_l1=L1_loss(fake,image_y)*100

        errG= errG_img +errG_vid + errG_l1

        errG.backward()
        D_G_z2 = output.mean().item()
        if opt.schedular == "True":
            schedulerG.step()
        else:
            optimizerG.step()

        #print("Eng netg")

        """
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        """

        train_d_img_loss += errD_img.item()
        train_d_vid_loss += (errD_fake_vid+errD_real_vid).item()

        train_g_img_loss += errG_img.item()
        train_g_vid_loss += errG_vid.item()

        train_g_l1_loss += errG_l1.item()

        train_gtotal_loss += errG.item()

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
            vutils.save_image(image_y[half:half+num_save,:,:,:],
                    '%s/train/real_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name,epoch,i),
                    normalize=True)
            vutils.save_image(image_y[0,:,:,:],
                    '%s/train/reference_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name,epoch,i),
                    normalize=True)
            vutils.save_image(fake[half:half+num_save,:,:,:],
                    '%s/train/fake_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name, epoch,i),
                    normalize=True)
        i+=1



    # do checkpointing
    with torch.no_grad():
        i=0
        for data in val_loader:
            ref_image,image_y,ref_landmark,landmarks = data


            image_y = image_y.to(device).view((-1,3,64,64))
            step_size = image_y.size(0)
            ref_image = ref_image.to(device)
            ref_landmark = ref_landmark.to(device)
            landmarks = landmarks.to(device).view((-1,1,64,64))


            label = torch.full((step_size//2,), real_label, device=device)
            vid_label = torch.full((1,), real_label, device=device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):


                random_frames = torch.randperm(step_size)[:step_size//2]

                #this line is for reference concat
                output = netD_img(torch.cat((image_y[random_frames,:,:,:],ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
                errD_img_real = criterion(output, label)
                val_real_img_acc+=(output.mean().item()>0.5)

                ################################################################
                # D_vid real
                ################################################################

                output_vid = netD_vid(image_y)
                errD_real_vid = criterion(output_vid, vid_label)
                val_real_vid_acc+=(output_vid.mean().item()>0.5)

                #noise = torch.randn(step_size,128, device=device)
                fake = netG(ref_image,ref_landmark,landmarks)

                #fake = augment(fake.squeeze(0))

                fake = fake.squeeze(0)
                label.fill_(fake_label)
                vid_label.fill_(fake_label)

                #######################################################
                #D_img loss
                #######################################################
                #this line is for reference concat
                output = netD_img(torch.cat((fake[random_frames,:,:,:],ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
                errD_img_fake = criterion(output, label)
                val_fake_img_acc+=(output.mean().item()<0.5)
                ################################################################
                # D_vid
                ################################################################
                output = netD_vid(fake)
                errD_fake_vid = criterion(output, vid_label)
                val_fake_vid_acc+=(output.mean().item()<0.5)

                #################################################################
                # G_loss
                ################################################################

                label.fill_(real_label)  # fake labels are real for generator cost
                vid_label.fill_(real_label)
                #this line is for reference concat
                output = netD_img(torch.cat((fake[random_frames,:,:,:],ref_image.repeat(step_size//2,1,1,1),ref_landmark.repeat(step_size//2,1,1,1),landmarks[random_frames,:,:,:]),1))
                errG_img = criterion(output, label)

                output = netD_vid(fake)
                errG_vid =  criterion(output, vid_label)

                errG_l1=L1_loss(fake,image_y)*100

                errG=errG_img+errG_vid+errG_l1
                errD_vid = errD_real_vid+errD_fake_vid
                errD_img = errD_img_fake+errD_img_real


            #test_l1_loss += l1_loss.item()
            val_d_img_loss += errD_img.item()
            val_d_vid_loss += errD_vid.item()

            val_g_img_loss += errG_img.item()
            val_g_vid_loss += errG_vid.item()

            val_g_l1_loss += errG_l1.item()

            val_gtotal_loss +=errG.item()



            frame_num_output=fake.size(0)
            half = frame_num_output//2
            if i<3:
                num_save = 20
                if half+num_save>frame_num_output:
                    num_save = 10

                vutils.save_image(image_y[half:half+num_save,:,:,:],
                        '%s/val/real_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name,epoch,i),
                        normalize=True)
                vutils.save_image(ref_image[0,:,:,:],
                        '%s/val/reference_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name,epoch,i),
                        normalize=True)
                vutils.save_image(fake[half:half+num_save,:,:,:],
                        '%s/val/fake_samples_epoch_%03d_seq%d.jpg' % ("Landmark_Img_folder/"+file_name, epoch,i),
                        normalize=True)
            i+=1

            # statistics
            """
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(test_loader),
                     errD.item(), errG.item(),  D_G_z1, D_G_z2))
            """

    data = {'epoch': epoch,
    'train_g_l1_loss': train_g_l1_loss/len(train_loader),
    'train_d_img_loss':train_d_img_loss/len(train_loader),
    'train_d_vid_loss':train_d_vid_loss/len(train_loader),
    'train_g_img_loss':train_g_img_loss/len(train_loader),
    'train_g_vid_loss':train_g_vid_loss/len(train_loader),
    'train_gtotal_loss':train_gtotal_loss/len(train_loader),
    'val_g_l1_loss':val_g_l1_loss/len(val_loader),
    'val_d_img_loss':val_d_img_loss/len(val_loader),
    'val_d_vid_loss':val_d_vid_loss/len(val_loader),
    'val_g_img_loss':val_g_img_loss/len(val_loader),
    'val_g_vid_loss':val_g_vid_loss/len(val_loader),
    'val_gtotal_loss':val_gtotal_loss/len(val_loader),
    'train_len':len(train_loader),
    'test_len':len(val_loader),
    "train_real_img_acc":train_real_img_acc/(len(train_loader)*2),
    "train_real_vid_acc":train_real_vid_acc/(len(train_loader)*2),
    "train_fake_img_acc":train_fake_img_acc/(len(train_loader)*2),
    "train_fake_vid_acc":train_fake_vid_acc/(len(train_loader)*2),

    "val_real_img_acc":val_real_img_acc/(len(val_loader)*2),
    "val_real_vid_acc":val_real_vid_acc/(len(val_loader)*2),
    "val_fake_img_acc":val_fake_img_acc/(len(val_loader)*2),
    "val_fake_vid_acc":val_fake_vid_acc/(len(val_loader)*2),
    }
    if opt.log == "True":
        my_file=open(plot_file, "a")
    df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
    df.to_csv(my_file, header=False,index=False)
    my_file.close()

    torch.save(netG.state_dict(), '%s/Landmark_Models/%s/netG_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netD_img.state_dict(), '%s/Landmark_Models/%s/netD_img_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netD_vid.state_dict(), '%s/Landmark_Models/%s/netD_vid_epoch_%d.pth' % (opt.outf, file_name,epoch))
    torch.save(netG.state_dict(), '%s/Landmark_Models/%s/netG_epoch_%d.pth' % (opt.outf, file_name,epoch))
    ##########################################################################################################
    #Save optimizers
    ##########################################################################################################
    torch.save(optimizerD_img.state_dict(), '%s/Landmark_Models/%s/optimizerD_img.pth' % (opt.outf, file_name))
    torch.save(optimizerD_vid.state_dict(), '%s/Landmark_Models/%s/optimizerD_vid.pth' % (opt.outf, file_name))
    torch.save(optimizerG.state_dict(), '%s/Landmark_Models/%s/optimizerG.pth' % (opt.outf, file_name))
