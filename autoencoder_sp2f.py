from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms,utils
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datetime  import datetime
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import h5py
import torchvision.utils as vutils
from torch.nn.utils.rnn import pad_sequence

np.random.seed(2)
torch.manual_seed(2)


class Net(nn.Module):
    def __init__(self,latent_dim):
        super(Net, self).__init__()
        self.latent_dim = latent_dim

        ngf = 64
        ndf = 64
        self.image_encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.latent_dim, 4, 1, 0, bias=False),
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

        self.audio_linear = nn.Linear(ndf*60*3,self.latent_dim)

        decoder_dim = self.latent_dim*2 + 4
        self.Decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     decoder_dim , ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )



    def forward(self,ref_image,audio_data,one_hot_label):

        batch_size = audio_data.size(0)
        x1 = self.image_encoder(ref_image)


        x2 = self.audio_encoder(audio_data.unsqueeze(1))

        x2 = self.audio_linear(x2.view(-1,64*60*3))


        z_latent = torch.cat((x1,x2.view(-1,self.latent_dim,1,1),one_hot_label.view(-1,4,1,1)),axis=1)

        out = self.Decoder(z_latent)

        return out


def get_label(name): #converts names to int labels
    if "ANG" in name:
        return 0
    elif "HAP" in name:
        return 1
    elif "NEU" in name:
        return 2
    elif "SAD" in name:
        return 3
count = 0

class Audio_video_dataset(Dataset):


    def __init__(self, h5_file, video_root_dir, transform=None):


        self.h5_file = h5_file

        self.audio_dict = {}

        self.frame_list = []

        with  h5py.File(self.h5_file, 'r') as f:
            self.keys = list(f.keys())
            for key in tqdm(self.keys):
                self.audio_dict[key]=f[key][:]
                self.frame_list = self.frame_list + [key+"/"+str(x) for x in range(2,self.audio_dict[key].shape[0]+1)] #Start from 2 since 1 is reference, add 1 to the endbcause of the lenght
        self.video_root_dir = video_root_dir
        self.transform = transform
        """
        video_folder_list = os.listdir(video_root_dir)
        self.video_list = []
        for i in tqdm(video_folder_list):
            self.video_list = self.video_list  + [i+"/"+x for x in os.listdir(os.path.join(video_root_dir,i))]
        """


    def __len__(self):
        return len(self.frame_list)


    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py
    """
    def read_images(self, video_dir):
        image_list=os.listdir(video_dir)

        X = []
        for i in range(1,len(image_list)+1):
            image = Image.open(os.path.join(video_dir,'image-{:d}.jpeg'.format(i)))

            if self.transform:
                image = self.transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X
    """

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.frame_list[idx] #current elemen (audio,video)

        #example frame_list element "1088_ITH_HAP_XX/41"
        key_name = selected_elem.split("/")[0]
        frame_num = int(selected_elem.split("/")[1])

        audio_data = torch.from_numpy(self.audio_dict[key_name][frame_num-1,:,:]) #images are from 1-... but arrays are 0-.. so -1

        label = get_label(key_name)

        sequence_name = os.path.join(self.video_root_dir,
                                key_name)

        image_y = Image.open(os.path.join(sequence_name,'image-{:d}.jpeg'.format(frame_num)))
        image_y = self.transform(image_y)

        ref_image = Image.open(os.path.join(sequence_name,'image-{:d}.jpeg'.format(1)))
        ref_image = self.transform(ref_image)

        return ref_image,image_y,audio_data,label

#https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html
#https://www.cs.virginia.edu/~vicente/recognition/notebooks/rnn_lab.html
#https://scikit-image.org/docs/dev/api/skimage.io.html

#transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])


dataset_train = Audio_video_dataset(h5_file="data/audio/audio_features_4class(1pad)_train.hdf5",
                                           video_root_dir='data/video/cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))
dataset_test = Audio_video_dataset(h5_file="data/audio/audio_features_4class(1pad)_test.hdf5",
                                           video_root_dir='data/video/cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))

batch_size = 256

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4)




test_loader = DataLoader(dataset_test, batch_size=batch_size,
                        shuffle=True, num_workers=4)


train_set_size = len(dataset_train)
test_set_size = len(dataset_test)

print("Train set size:",train_set_size)
print("Test set size:",test_set_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)

"""
def forward(self, x):

    batch_size, timesteps, C,H, W = x.size()
    c_in = x.view(batch_size * timesteps, C, H, W)

    c_out = self.cnn(c_in)

    r_out, (h_n, h_c) = self.rnn(c_out.view(-1,batch_size,c_out.shape[-1]))

    logits = self.classifier(r_out)

    return logits
"""
#class_names = dataset_train.classes






# Get a batch of training data

ref_image,image_y, audio_data ,labels = next(iter(train_loader))
print("Ref imsize",ref_image.shape)
print("Label imsize",image_y.shape)
print("Audio input size",audio_data.shape)


def train_model(model, criterion, optimizer,exp_lr_scheduler=None,num_epochs=25,checkp_epoch=0):
    since = time.time()

    
    my_file=open(plot_file, "a")


    pbar=tqdm(range(checkp_epoch,num_epochs))
    for epoch in pbar: #range(checkp_epoch,num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        #pbar2=tqdm(train_loader)
        for sample in train_loader:
            ref_image,image_y, audio_data ,labels = sample

            batch_size = audio_data.size(0)
            ref_image = ref_image.to(device)
            image_y = image_y.to(device)
            one_hot_label = torch.zeros(batch_size,4).to(device) #number of class
            one_hot_label[range(batch_size),labels]=1
            audio_data = audio_data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):

                outputs = model(ref_image,audio_data,one_hot_label)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, image_y)


                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * image_y.size(0)


        train_loss = running_loss / train_set_size


            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
        model.eval()   # Set model to evaluate mode

        running_loss = 0.0


        with torch.no_grad():
            i=0
            for sample in test_loader:
                ref_image,image_y, audio_data ,labels = sample
                batch_size = audio_data.size(0)
                ref_image = ref_image.to(device)
                image_y = image_y.to(device)
                one_hot_label = torch.zeros(batch_size,4).to(device) #number of class
                one_hot_label[range(batch_size),labels]=1
                audio_data = audio_data.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model(ref_image,audio_data,one_hot_label)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs,image_y)

                if i==0:
                    num_save = 32
                    vutils.save_image(image_y[:num_save,:,:,:],
                            '%s/real_samples_epoch_%03d.png' % ("Img_folder",epoch),
                            normalize=False)
                    vutils.save_image(ref_image[:num_save,:,:,:],
                            '%s/reference_samples_epoch_%03d.png' % ("Img_folder",epoch),
                            normalize=False)
                    vutils.save_image(outputs.detach()[:num_save,:,:,:],
                            '%s/fake_samples_epoch_%03d.png' % ("Img_folder", epoch),
                            normalize=False)
                i+=1

                # statistics
                running_loss += loss.item() * image_y.size(0)


        test_loss = running_loss / test_set_size


        torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss
             },checkpoint_file+str(epoch)+".pt")


        data = {'epoch': epoch,
        'train_loss': train_loss,
        'test_loss':test_loss
        }
        df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
        df.to_csv(my_file, header=False,index=False)
        #print()

        pbar.set_description("train loss {:.4} test loss {:.4}".format(train_loss,test_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # load best model weights

    return model



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model = Net(1024).to(device)

model.apply(weights_init)

criterion = nn.MSELoss()


optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


now=datetime.now()

checkpoint_file="./Models/auto_encode"
plot_file="auto_encode"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


num_epochs=100

if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        model = train_model(model, criterion, optimizer,
                       num_epochs=num_epochs,checkp_epoch=checkpoint['epoch']+1)
else:
    model = train_model(model, criterion, optimizer,
                       num_epochs=num_epochs)
