import os
import time

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import segmentation_models_pytorch as smp
from SMI980.fill_hole import qu_bai, qu_hei
from segmentation.light_Net.ContextNet import ContextNet
from segmentation.light_Net.DABNet import DABNet
from segmentation.light_Net.EDANet import EDANet
from segmentation.light_Net.ESNet import ESNet
from segmentation.light_Net.FPENet import FPENet
from segmentation.light_Net.FastSCNN import FastSCNN
from segmentation.light_Net.IRDP import IRDPNet
from segmentation.light_Net.UNet import UNet
from zhuan_L_image import convert_rgb_to_grayscale
from segmentation.light_Net.LETNet import LETNet
from segmentation.FCN import VGGNet,FCNs
from segmentation.HRNet import HighResolutionNet
from segmentation.Evaluation import evaluate

# =========================== 数据集类 ===========================
class MyData(Dataset):
    def __init__(self, img_dir ):
        self.img_dir = img_dir
        self.img_names = os.listdir(self.img_dir)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


# =========================== 主逻辑 ===========================
def main(save_path= r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\segmentation\mydata\our\pre2" ):
    model_name=os.path.basename(os.path.dirname(save_path) )
    root_dir = r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\ggb_new_25_1021\open_data_ic_from_chengderuo\test\image"
    # input_subdir = "image"
    model_path = r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\segmentation\our_cd_251102_RGB_23best_model25_0623.pth"

    # model_path =r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\segmentation\LETNET_251022_RGB_12best_model25_0623.pth"
    os.makedirs(save_path, exist_ok=True)
    # 数据加载
    dataset = MyData(root_dir)
    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if model_name in ["EUNet++"]:
    # for model_name in model_list_name:
    if model_name in ["EUNet++","EUnetPlusPlus"]:
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1  # model output channels (number of classes in your dataset)
        ).cuda()
    if model_name in ["FPENet"]:
        model = FPENet(in_channels=1, classes=1).cuda()
    if model_name in ["our"]:
        model = DABNet(in_channels=1, classes=1).cuda()
    if model_name in ["FCN"]:
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        model = FCNs(pretrained_net=vgg_model, n_class=1).cuda()
    if model_name in ["IRDPnet"]:
        model = IRDPNet(in_channels=1, classes=1).cuda()
    if model_name in ["HRNet"]:
        model = HighResolutionNet(in_channel=1, num_classes=1).cuda()
    if model_name in ["ESNet"]:
        model = ESNet(in_channels=1, classes=1).cuda()
    if model_name in ["EDAnet"]:
        model = EDANet(in_channels=1, classes=1).cuda()
    if model_name in ["contextNet"]:
        model = ContextNet(in_channels=1, classes=1).cuda()
    if model_name in ["UNET"]:
        model = UNet(in_channels=1, classes=1).cuda()
    if model_name in ["LETnet"]:
        from segmentation.light_Net.LETNet import LETNet
        model = LETNet(inchanel=1, classes=1).cuda()
    if model_name in ["MAnet"]:
        model = smp.MAnet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1  # model output channels (number of classes in your dataset)
        ).cuda()
    # 图像预测
    transform = transforms.ToTensor()
    processed_images = set(os.listdir(save_path))
    idx_end=0
    for idx, (img_pil, name) in enumerate(dataset, start=1):
        # if name in processed_images:
        #     print(f"[{idx:03d}] Skipping already processed image '{name}'")
    # for idx, (img_pil, name) in enumerate(dataset, start=1):
        start_time = time.time()
        img_tensor = transform(img_pil).unsqueeze(0).to(device)


        with torch.no_grad():
            model.load_state_dict(
                torch.load(model_path))
            output = model(img_tensor)[0]  # shape: [1, 1024, 1024]
            print(output.shape)

        save_image(output, os.path.join(save_path, name))

        elapsed = time.time() - start_time
        print(f"[{idx:03d}] Processed '{name}' in {elapsed:.2f} s")
        idx_end=idx
    return save_path,idx_end

if __name__ == '__main__':
    # save_path= r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\ggb_new_25_1021\train\val\pre_DABNET"
    start_time=time.time()
    save_path,idx_end=main()
    # print(f"<<<<<<<所花时间{round((start_time - time.time())*(100/idx_end), 3)}>>>>>>>>：")
    label_dir=r"D:\python_new\ggb_new_25_1021\hainan_and_segmentation_251021\ggb_new_25_1021\open_data_ic_from_chengderuo\test\label"
    evaluate(label_dir, save_path, threshold=128)

