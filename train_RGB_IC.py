import time
import torchvision
from torch import nn
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from segmentation.HRNet import HighResolutionNet

import numpy as np
import segmentation_models_pytorch as smp
from segmentation.light_Net.ContextNet import ContextNet
from segmentation.light_Net.EDANet import EDANet
from segmentation.light_Net.ESNet import ESNet
from segmentation.light_Net.FPENet import FPENet
from segmentation.light_Net.FastSCNN import FastSCNN
from segmentation.light_Net.IRDP import IRDPNet
from segmentation.light_Net.LETNet import LETNet
from EAL_ICNet import EAL_ICNet
from segmentation.FCN import VGGNet,FCNs
from segmentation.light_Net.UNet import UNet


tran=transforms.ToTensor()

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss (TransAttUnet-style)
    L = α * L_BCE + β * L_Dice
    """

    def __init__(self, alpha=0.9, beta=0.1, epsilon=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()  # 与论文相同：使用 logit 输入的 BCE

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)

        # Sigmoid 激活得到概率
        probs = torch.sigmoid(inputs)

        # Dice Loss (基于论文公式)
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice_loss = 1 - (2. * intersection + self.epsilon) / (union + self.epsilon)

        # 加权组合
        total_loss = self.alpha * bce_loss + self.beta * dice_loss

        return total_loss
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience  # 容忍多少轮没有提升
        self.min_delta = min_delta  # 最小提升量
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"早停计数器：{self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class Dataset1(Dataset):
    # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
    def __init__(self, root_dir,value_dir,label_dir,h_and_w):
        self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
        self.label_dir = label_dir
        self.value_dir=value_dir
        self.path2= os.path.join(self.root_dir, self.label_dir) # 路径的拼接
        self.path1= os.path.join(self.root_dir, self.value_dir) # 路径的拼接
        self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
        self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址
        self.h_and_w=h_and_w
    ## 获取所有图片的地址列表
    def __getitem__(self, idx):
        img_name1 = self.img_path1[idx] #获取图片名称  self.全局的
        img_name2= self.img_path2[idx] #获取图片名称  self.全局的
        img_item_path = os.path.join(self.path1, img_name1) # 获取每个图片的地址(相对路径)
        smg_item_path = os.path.join(self.path2, img_name2) # 获取每个图片的地址(相对路径)
        img1= Image.open(img_item_path)
        img2=Image.open(smg_item_path)
        img1=tran(img1)
        img2=tran(img2)
        # print(img1.shape)
        # print(img2.shape)
        # img1=img1[0]
        # img2= img2[0]
        # img1=torch.reshape(img1,(1,self.h_and_w,self.h_and_w))
        # img2=torch.reshape(img2,(1,self.h_and_w,self.h_and_w))
        return img1,img2

    def __len__(self):
        return len(self.img_path1)#这里返回一个就行



class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self,x:torch.Tensor):
        return self.pool(x)
class UpSample(nn.Module):
    def __init__(self,input_channals:int,output_channals:int):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channals,output_channals,kernel_size=2,stride=2)
        #看效果，不好试试UpsamplingBilinear2d(scale_factor=2)
    def forward(self,x:torch.Tensor):
        return self.up(x)
class CropAndConcat(nn.Module):

    def forward(self,x:torch.Tensor,contracting_x:torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x,[x.shape[2],x.shape[3]])
        x = torch.cat([x,contracting_x],dim=1)
        return x

def train_process(model, data_train, data_test,model_name,num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = BCEDiceLoss(alpha=0.5, beta=0.5, epsilon=1e-6)
    model = model.to(device)
    # 复制当前模型的参数
    # 训练集损失函数的列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 计时(当前时间)
    since = time.time()
    k=1
    best_loss = float('inf')
    #squeeze函数会把前面四则（1，*，*）变为（*，*）
    early_stopping = EarlyStopping(patience=5, min_delta=0.0000001)
    for i in range(num_epoch):
        print(f'第{k}轮开始,总共{num_epoch}轮')
        #初始化值
        train_loss=0
        #训练集准确度
        train_correct=0
        val_loss=0
        #验证集的准确度
        val_correct=0
        train_num=0
        val_num=0
        plt.figure(figsize=(12,5))
        for batch_idx, (image,label) in enumerate(tqdm(data_train, desc="训练中")):
            image=image.to(device)
            label=label.to(device)
            #训练模式
            model.train()
            output=model(image)
            # pre_label=torch.argmax(output,dim=1)
            loss_train=loss(output,label)
            optim.zero_grad()
            #这里的loss_train为64个样本的平均值
            loss_train.backward()

            optim.step()
            train_loss+=loss_train.item()*image.size(0)#总的样本loss的累加
            train_correct+=torch.sum(output==label)
            train_num+=image.size(0)
        for jp in data_test:
            image1,label1=jp
            image1=image1.to(device)
            label1=label1.to(device)
            #评估模式
            model.eval()
            output1=model(image1)
            # pre_label_test=torch.argmax(output,dim=1)
            loss_test=loss(output1,label1)
            #对损失函数进行累加
            val_loss+=loss_test.item()*image.size(0)#这里乘以64了
            val_correct+=torch.sum(output==label)
            val_num+=image.size(0)

        #该轮次平均的loss
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)

        if val_loss_all[-1]<best_loss:
            best_loss=val_loss_all[-1]
            print("best_loss=",best_loss)
            #保存参数
            best_acc_wts=copy.deepcopy(model.state_dict())
            torch.save(best_acc_wts, f'{model_name}_{k}best_model25_0623.pth')
        #时间
        # 添加 EarlyStopping 检查
        early_stopping(val_loss_all[-1])
        if early_stopping.early_stop:
            print("验证集损失无提升，触发早停！")
            break
        time_use=time.time()-since
        print(f'训练总耗费时间{time_use//60}m,{time_use%60}s')
        k+=1
    #选择最优参数
    #选择最高精确度的模型参数
    torch.save(best_acc_wts,fr'{model_name}_25_.pth')


if __name__ == '__main__':
    data = Dataset1(r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\ggb_new_25_1021\open_data_ic_from_chengderuo\train", "image", "label", h_and_w=512)
    data2 = Dataset1(r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\ggb_new_25_1021\open_data_ic_from_chengderuo\val", "image", "label", h_and_w=512)
    data2_loader = DataLoader(data2, batch_size=1, shuffle=True)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    # model_list_name=["FPENet","our","FCN","IRDPnet","HRNet","ESNet","EDAnet","UNet","EUNet++","LETnet"]
    model_list_name=["our"]
    model = None
    for model_name in model_list_name:
        if model_name in ["EUNet++"]:
            model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b0",
                encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1  # model output channels (number of classes in your dataset)
            )
        if model_name in ["FPENet"]:
            model = FPENet(in_channels=1,classes=1)
        if model_name in ["our"]:
            model=EAL_ICNet(in_channels=1,classes=1)
        if model_name in ["FCN"]:
            vgg_model = VGGNet(requires_grad=True, show_params=False)
            model = FCNs(pretrained_net=vgg_model, n_class=1)
        if model_name in ["IRDPnet"]:
            model = IRDPNet(in_channels=1,classes=1)
        if model_name in ["HRNet"]:
            model = HighResolutionNet(in_channel=1, num_classes=1)
        if model_name in ["ESNet"]:
            model = ESNet(in_channels=1,classes=1)
        if model_name in ["EDAnet"]:
            model = EDANet(in_channels=1,classes=1)
        if model_name in ["contextNet"]:
            model = ContextNet(in_channels=1,classes=1)
        if model_name in ["UNet"]:
            model = UNet(in_channels=1,classes=1)
        if model_name in ["LETnet"]:
            from segmentation.light_Net.LETNet import LETNet
            model = LETNet(inchanel=3, classes=1)
        if model_name in ["MAnet"]:
            model = smp.MAnet(
                encoder_name="efficientnet-b0",
                encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1  # model output channels (number of classes in your dataset)
            ).cuda()
        print(f"_____________{model_name}开始训练______________")
        train_process(model,data_loader,data2_loader,model_name=f'{model_name}_cd_251102_RGB',num_epoch=30)

