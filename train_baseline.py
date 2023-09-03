from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    AddChanneld,
    SpatialPadd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    EnsureType,
    Invertd,
    ToTensord,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, DynUNet, SegResNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, DiceFocalLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch,load_decathlon_datalist
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import tempfile
import shutil
import os
import glob
from random import shuffle
import pickle
import numpy as np
import torch.multiprocessing
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

set_determinism(seed=0)
device = torch.device("cuda:2")
data_dir = os.path.normpath('/data/liucd/Dataset/LungSeg/')

all_images = sorted(glob.glob(os.path.join(data_dir, 'imagesTr', '*.nii.gz')))
all_labels = sorted(glob.glob(os.path.join(data_dir, 'labelsTr', '*.nii.gz')))
all_dict = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(all_images, all_labels)]
split = int(0.8 * len(all_dict))
train_dict = all_dict[:split]
val_dict = all_dict[split:]
print(len(train_dict), len(val_dict))

logdir = os.path.normpath('./logs')

model_dir = './'
best_metric_model_path = 'demo.pth'


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
     def __call__(self, data):
        

        x = data['label']
        if list(np.unique(x)) == [0, 2]:
            x[x > 1] = 1
        if list(np.unique(x)) == [0, 1, 2]:
            x[x == 2] = 0

        data['label'] = x

        return data


train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2, 2, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024, a_max=600,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=3,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            
            ScaleIntensityRanged(keys=["image"],a_min=-1024, a_max=600,b_min=0.0, b_max=1.0,clip=True ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["image", "label"]),
        ]
    )

train_ds = CacheDataset(
        data=train_dict,
        transform=train_transforms,
        #cache_num=6,
        cache_rate=1.0,
        num_workers=24,
    )
train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=24, pin_memory=True
    )
val_ds = CacheDataset(
        data=val_dict,
        transform=val_transforms,
        #cache_num=6,
        cache_rate=1.0,
        num_workers=4
    )
val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

print(len(train_ds), len(val_ds))
max_epochs = 500
val_interval = 2


model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.2,
).to(device)


torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-6)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean_batch")  
loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice = 1, lambda_ce = 1)   
# loss_function = DiceFocalLoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.1, lambda_focal=0.9)   


best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
# 以下两步后处理只针对验证集
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])  # argmax将predict的channel维度的结果变为1，to_onehot又将其变为cls维度的向量
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])   # label的channel维度本身就是1(EnsureChannelFirstd带来的)，所以只需要to_onehot

for epoch in range(max_epochs):
    print(f"epoch {epoch + 1}/{max_epochs}", end=' ')
    
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )  # shape --> input: torch.Size([8, 1, 96, 96, 96]) label: torch.Size([8, 1, 96, 96, 96])  
        
        optimizer.zero_grad()
        outputs = model(inputs)  # shape: torch.Size([8, 2, 96, 96, 96])
        # print(outputs.max().item(), outputs.min().item())  # 21.267742156982422 -8.954521179199219 
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
#         lr_scheduler.step()
        epoch_loss += loss.item()
        # print(
        #    f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #    f"train_loss: {loss.item():.4f}")
        
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )   # input: torch.Size([1, 1, 226, 157, 113])  label: torch.Size([1, 1, 226, 157, 113])
                roi_size = (96, 96, 96)
                sw_batch_size = 4  # 这里的sw_batch_size 应该是slide总数以batch为4的小图进行推理，而不是取验证集里的4张大图进行推理
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)   # val_outputs: torch.Size([1, 2, 226, 157, 113])
                # 以下两个变换保证 output/label均为列表形式，列表里的每个元素都是channel,x,y,z的格式
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]  
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)  # 
                
            # aggregate the final mean dice result
            metric = dice_metric.aggregate() # [dice_background, dice_liver, dice_tumor]
            dice_metric.reset()
            
            
            metric_values.append(metric[1])
            if metric[1].item() > best_metric:  # 肝脏分割最好的模型被保存
                best_metric = metric[1].item()
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    model_dir, best_metric_model_path))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current lung dice: {metric[1].item():.4f}"
        
                f" best mean dice: {best_metric:.4f} "
                f" at epoch: {best_metric_epoch}"
            )
