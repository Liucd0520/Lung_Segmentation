from tqdm import tqdm
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
    EnsureTyped,
    EnsureType,
    ToTensord,
    Invertd,
    NormalizeIntensityd,
    RandFlipd, 
    AddChanneld,
    SpatialPadd,
    Lambdad,
    MapTransform,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, DynUNet, UNETR, SegResNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, DiceFocalLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import tempfile
import shutil
import os
import glob
import pickle 
import numpy as np
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


device = torch.device("cuda:1")

data_dir = os.path.normpath('../test_image')
all_images = sorted(glob.glob(os.path.join(data_dir,  '*.nii.gz')))
val_dict = [{'image': image_name} for image_name in all_images]

logdir = os.path.normpath('./logs')


pretrained_path = os.path.join('demo.pth')



val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", ]),
        AddChanneld(keys=["image", ]),
        Orientationd(keys=["image",], axcodes="RAS"),
        Spacingd(
                keys=["image", ],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear",),
            ),

        
        ScaleIntensityRanged(keys=["image"],a_min=-1024, a_max=600,b_min=0.0, b_max=1.0,clip=True ),
        CropForegroundd(keys=["image", ], source_key="image"),
        SpatialPadd(keys=["image", ], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image",]),
    ]
)

val_ds = CacheDataset(
        data=val_dict,
        transform=val_org_transforms,
        #cache_num=6,
        cache_rate=1.0,
        num_workers=4
    )


val_org_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

print(len(val_ds))


post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=val_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
])


model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.2,
).to(device)


model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cuda:1')))

torch.backends.cudnn.benchmark = True


save_pred_trans = Compose([
    Lambdad(keys=['pred'], func=lambda x: x[1:, ...]),
    SaveImaged(keys=['pred'], output_dir='./test_pred', output_postfix='', separate_folder=False)
])  # x[1:, ...] 保证shape是[1, x, x, x],才能被保存


post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        liucds = []
        with torch.no_grad():
            for step, val_data in enumerate(epoch_iterator_val):
                val_inputs = val_data["image"].to(device)
                print(val_inputs.shape)
                
                val_data["pred"] = sliding_window_inference(val_inputs, (96, 96, 96), 12, model)
      
    
                val_data = [post_transforms(i) for i in decollate_batch(val_data)]
                save_pred_trans(val_data[0]) # batch_size=1,所以是val_data[0]
                

epoch_iterator_val = tqdm(
                    val_org_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )

validation(epoch_iterator_val)




