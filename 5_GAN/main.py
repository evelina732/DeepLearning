# -*- coding: utf-8 -*-
"""lab5 _310605022.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f2UHv96Xa5jOlU52hnINvff4RIe9KJuC
"""

import gdown
import json
import cv2
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms as transforms

import PIL
from PIL import Image

"""## prepare"""

from google.colab import drive
drive.mount('/content/drive')

import gdown

url = "https://drive.google.com/uc?id=1y1x5aZjQR31IHKnXqRSqpNbOdDvIxXRc&export=download"
gdown.download(url, "iclevr.zip")

!mkdir ./data
!cp /content/iclevr.zip  ./data/
!cd ./data && unzip iclevr.zip

"""## dataloader"""

def getData(mode):
    f = open("/content/data/objects.json")
    object_dict = json.load(f)

    if mode == 'train':
        f = open("/content/data/train.json")
        train_dict = json.load(f)
        train_name = list(train_dict.keys())
        train_label = list(train_dict.values())
        for i in range(len(train_label)):
            one_hot_arr = np.zeros((24, 1))
            for j in train_label[i]:
                one_hot_arr[object_dict[j]] = 1
            train_label[i] = np.squeeze(one_hot_arr)

        return train_name, train_label
    else:
        f = open("/content/data/test.json")
        test_label = json.load(f)
        for i in range(len(test_label)):
            one_hot_arr = np.zeros((24, 1))
            for j in test_label[i]:
                one_hot_arr[object_dict[j]] = 1
            test_label[i] = np.squeeze(one_hot_arr)

        return test_label

class DataSet():

    def __init__(self, root, mode, transform=None):

        self.root = root
        self.transform = transform
        self.mode = mode

        if mode == "train":
            self.img_name, self.img_label = getData(mode)
        elif mode == "test":
            self.img_label = getData(mode)


        print("> Found %d images..." % (len(self.img_label)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_label)

    def __getitem__(self, index):
        if self.mode == "train":
            # print(self.root + "/" + self.img_name[index])
            path_list = self.root + "/" + self.img_name[index]
            image = cv2.imread(path_list)

            if self.transform:
                image = self.transform(image)
            # image = np.transpose(image, (1, 2, 0))

            label = self.img_label[index]

            return torch.Tensor(image), torch.Tensor(label)

        else:
            label = self.img_label[index]

            return torch.Tensor(label)

"""## train function

"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=9, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")

parser.add_argument('--ngf', type=int, default=300, help="feature channels of generator")
parser.add_argument('--ndf', type=int, default=100, help="feature channels of discriminator")
parser.add_argument("--nc", type=int, default=100, help="number of condition embedding dim")


args = parser.parse_args(args=[])

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        self.ngf, self.nc, self.nz = args.ngf, args.nc, args.latent_dim
        self.n_classes = args.n_classes

        # condition embedding
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.nc),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz + self.nc, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, self.nc, 1, 1)
        gen_input = torch.cat((label_emb, noise), 1)
        out = self.main(gen_input)
        return out

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ndf, self.nc = args.ndf, args.nc
        self.n_classes = args.n_classes
        self.img_size = args.img_size
        self.main = nn.Sequential(
            # input is (rgb chnannel = 3) x 64 x 64
            nn.Conv2d(3, self.ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (ndf*16) x 8 x 8
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

        )

        # discriminator fc
        self.fc_dis = nn.Sequential(
            nn.Linear(6*6*self.ndf*32, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc
        self.fc_aux = nn.Sequential(
            nn.Linear(6*6*self.ndf*32, self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv = self.main(input)
        flat = conv.view(-1, 6*6*self.ndf*32)
        fc_dis = self.fc_dis(flat).view(-1, 1).squeeze(1)
        fc_aux = self.fc_aux(flat)
        return fc_dis.view(-1,1), fc_aux

"""##train"""

# Commented out IPython magic to ensure Python compatibility.
os.makedirs("images", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator(args)
discriminator = Discriminator(args)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# ---------------- Dataloader ----------------- #
input_root = "/content/data"

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((64, 64)),
     transforms.ToTensor()])

test_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((64, 64)),
     transforms.ToTensor()])

train_dataset= DataSet(input_root, "train", transform=train_transform)
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset= DataSet(input_root, "test", transform=test_transform)
test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim, 1, 1))))
    # genarator label

    gen_label = [None] * args.batch_size
    for j_ in range(args.batch_size):
        item_n = np.random.randint(1,4)
        one_hot_ = np.zeros((24, 1))
        for i_ in range(item_n):
            class_ = np.random.randint(0, args.n_classes, 1)
            one_hot_[class_] = 1
        gen_label[j_] = np.squeeze(one_hot_)
    gen_label = Variable(FloatTensor(gen_label))

    gen_imgs = generator(z, gen_label)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

evaluator = evaluation_model()
best_acc = 0
for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(train_dataloader):
        # for i, (imgs, labels) in enumerate(ic_train_dataloader):
        #     print()

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        real_labels = Variable(labels.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------
        discriminator.train()
        generator.train()
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim, 1, 1))))

        # genarator label
        gen_label = [None] * args.batch_size
        for j_ in range(args.batch_size):
            item_n = np.random.randint(1,4)
            one_hot_ = np.zeros((24, 1))
            for i_ in range(item_n):
                class_ = np.random.randint(0, args.n_classes, 1)
                one_hot_[class_] = 1
            gen_label[j_] = np.squeeze(one_hot_)
        gen_label = Variable(FloatTensor(gen_label))



        # Generate a batch of images
        if np.random.random() > 0.5:
            cond = real_labels
        else:
            cond = gen_label

        gen_imgs = generator(z, cond)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, cond))


        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, real_labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, cond)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()
        discriminator.eval()
        generator.eval()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
#             % (epoch, args.n_epochs, i, len(train_dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % args.sample_interval == 0:
            sample_image(n_row=8, batches_done=batches_done)


        with torch.no_grad():
            for iii, cond in enumerate(test_dataloader):
                cond =  Variable(cond.type(FloatTensor)).to('cuda')

                batch_size = cond.size(0)
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim, 1, 1)))).to('cuda')
                fake_image = generator(z, cond)
                save_image(fake_image.detach(),
                    'fake_test_samples_epoch_%03d.png' % (epoch),
                    normalize=True)
                acc = evaluator.eval(fake_image, cond)
                # do checkpointing
                if acc > best_acc:
                    best_acc = acc
                    torch.save(generator.state_dict(), 'netG.pth')
                    torch.save(discriminator.state_dict(), 'netD.pth')
        print("best: ", best_acc)
        print(i)

for iii, cond in enumerate(test_dataloader):
  print(cond.shape)

for data in train_dataloader:
    print(data)

trainX, _ = train_dataloader





import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:

DLP summer 2022 Lab5 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('/content/checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc




