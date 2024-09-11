import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from DAAUNet import *
import lovasz_losses as L
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path ='atd_net_spec_cirrus_spec_top723.pth'
train_data_path = r'D:\pw\x1\DAA-UNet\data'

save_path = 'train_image'

class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss

class meandiceloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        _,_,_,_,t=meandice(pred, label)
        loss1= 1 - t
        return loss1

def meandice(pred, label):
    sumdice = 0
    sumdice0 = 0
    sumdice1 = 0
    sumdice2 = 0
    sumdice3 = 0
    smooth = 1e-6
    for i in range(0, 4):
        pred_bin = (pred == i) * 1
        label_bin = (label == i) * 1
        pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
        label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)
        intersection = (pred_bin * label_bin).sum()
        dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
        if i == 0:
            sumdice0=dice
            sumdice += sumdice0
        elif i==1:
            sumdice1 = dice
            sumdice += sumdice1
        elif i==2:
            sumdice2 = dice
            sumdice += sumdice2
        elif i==3:
            sumdice3 = dice
            sumdice += sumdice3

    return sumdice0,sumdice1,sumdice2,sumdice3,sumdice / 4


if __name__ == '__main__':

    data_loader=MyDataset(train_data_path)
    import torch.utils.data as Data
    train_data_loader, val_data_loader = Data.random_split(data_loader,
                                                lengths=[int(0.8 * len(data_loader)),
                                                         len(data_loader) - int(0.8 * len(data_loader))],
                                                generator=torch.Generator().manual_seed(0))

    train_data_loader = DataLoader(train_data_loader, batch_size=16, shuffle=True)
    val_data_loader = DataLoader(val_data_loader, batch_size=4, shuffle=True)


    net = DAAUNet(img_ch=1, output_ch=4).to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    lr = 0.0001
    learning_rate_decay_start = 20
    learning_rate_decay_every = 10
    learning_rate_decay_rate = 0.9

    opt=optim.RMSprop(net.parameters(), lr=lr, eps=1e-8, weight_decay=1e-4)

    loss_fun=LovaszLossSoftmax()
    diceloss=meandiceloss()

    max_dice = 0
    max_epoch = 0
    epoch = 1

    while epoch < 200:
        train_num_correct = 0
        train_num_pixels = 0
        val_num_correct = 0
        val_num_pixels = 0

        train_Dice0 = 0
        train_Dice1 = 0
        train_Dice2 = 0
        train_Dice3 = 0
        train_Dice = 0

        train_Iou = 0

        val_Dice0 = 0
        val_Dice1 = 0
        val_Dice2 = 0
        val_Dice3 = 0
        val_Dice = 0

        val_Iou = 0
        train_cnt = 0
        val_cnt = 0

        net.train()

        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)

            prediction = torch.argmax(out_image, dim=1)
            td0,td1,td2,td3,rawdice = meandice(prediction, segment_image)

            train_Dice0 += td0
            train_Dice1 += td1
            train_Dice2 += td2
            train_Dice3 += td3
            train_Dice += rawdice

            train_loss = loss_fun(out_image, segment_image)*0.7+diceloss(prediction, segment_image)*0.3

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_cnt +=1

        print(f'{epoch}-{i}-train_Dice===>>{train_Dice0/train_cnt:.4f},{train_Dice1/train_cnt:.4f},{train_Dice2/train_cnt:.4f},{train_Dice3/train_cnt:.4f},{train_Dice/train_cnt:.4f}')

        net.eval()
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)
                # print(out_image.shape)
                val_loss = loss_fun(out_image, segment_image.long())

                prediction = torch.argmax(out_image,  dim=1)
                vd0,vd1,vd2,vd3,rawdice = meandice(prediction, segment_image)
                val_Dice0+=vd0
                val_Dice1+=vd1
                val_Dice2+=vd2
                val_Dice3+=vd3
                val_Dice += rawdice

                val_cnt += 1

            print(f'{epoch}-{i}-val_Dice===>>{val_Dice0 / val_cnt:.4f},{val_Dice1 / val_cnt:.4f},{val_Dice2 / val_cnt:.4f},{val_Dice3 / val_cnt:.4f},{val_Dice / val_cnt:.4f}')
            # #print(f'{epoch}-{i}-val_IOU===>>{val_Iou / val_cnt}')
            if val_Dice / val_cnt > max_dice:
                max_dice = val_Dice / val_cnt
                max_epoch = epoch
                print('--------------------max_dice=', max_dice)
                print('--------------------max_epoch=', max_epoch)

                torch.save(net.state_dict(), weight_path)
                print('save successfully!')

        epoch += 1



