# Image-Adaptive-3DLUT

## Dataset and paper
### [Paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf), [Supplementary](https://www4.comp.polyu.edu.hk/~cslzhang/paper/Supplement_LUT.pdf)

Skin color correction dataset s3 bucket
`s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/dehaze_dataset`

## Usage
### Requirements
- instance: `g5.2xlarge`
```bash 
conda create -y -n 3dlut_train python=3.8
conda activate 3dlut_train 
pip install numpy==1.19.2 Pillow==6.1.0 opencv-python==3.4.8.29 scipy tqdm matplotlib 
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# torch 1.12 is required for onnx min() and max() conversion 
# pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

### Training
#### Download dataset from S3 
```bash
# size ~3GB
aws s3 sync s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/dehaze_dataset ./dataset
```
#### Train 
```bash
data_dir=./dataset
out=train_exp0

# initial train 
python image_adaptive_lut_train_paired.py \
--im_train ${data_dir}/train_im.txt \
--gt_train ${data_dir}/train_gt.txt \
--im_val ${data_dir}/val_im.txt \
--gt_val ${data_dir}/val_gt.txt \
--n_epochs 40 \
--output_dir LUTs/paired/${out} \
--lr 0.0001 \
--lambda_smooth 0.0001 \
--lambda_monotonicity 0 

# finetune with loss weight 
python image_adaptive_lut_train_paired.py \
--im_train ${data_dir}/train_im.txt \
--gt_train ${data_dir}/train_gt.txt \
--im_val ${data_dir}/val_im.txt \
--gt_val ${data_dir}/val_gt.txt \
--n_epochs 50 \
--output_dir LUTs/paired/${out} \
--lr 0.0001 \
--lambda_smooth 0.0001 \
--lambda_monotonicity 10 \
--epoch 39 \
--loss_weight 3
```

### Infer 
```bash
out=train_exp0
model_dir=$(pwd)/saved_models/LUTs/paired/${out}_sRGB
ep=49
ln -sf $model_dir/LUTs_${ep}.pth $model_dir/LUTs.pth
ln -sf $model_dir/classifier_${ep}.pth $model_dir/classifier.pth


### compute average CNN weights of LUTs, modify demo_eval.py 


im=demo_images/ENvsAF_clip1_540p_results_dehaze_composition_00000035.png
python demo_eval.py --image_name $im --model_dir $model_dir --output_dir ./ --result_name_suffix "_${out}_ep${ep}"
```

### Prepare for ONNX conversion 
- Compute average CNN weights to combine the 3 LUTs 
- Copy pred weights to dehaze_e2e.py to generate onnx and trt models 
```bash
data_dir=./dataset
python demo_eval.py --image_name ${data_dir}/train_im.txt --model_dir $model_dir --output_dir ${model_dir}/train_ep49 --result_name_suffix "_ep${ep}"

import numpy as np
preds = np.load('dataset_train_im.txt.npy')
preds.mean(axis=0)
```

### Tools
1. You can generate identity 3DLUT with arbitrary dimension by using `utils/generate_identity_3DLUT.py` as follows:

```
# you can replace 33 with any number you want
python3 utils/generate_identity_3DLUT.py -d 33
```

2. You can visualize the learned 3D LUT either by using the matlab code in `visualization_lut` or using the python code `utils/visualize_lut.py` as follows:

```
python3 utils/visualize_lut.py path/to/your/lut
# you can also modify the dimension of the lut as follows
python3 utils/visualize_lut.py path/to/your/lut --lut_dim 64
```
