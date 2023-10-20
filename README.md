# Image-Adaptive-3DLUT

## Dataset and paper
### [Paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf), [Supplementary](https://www4.comp.polyu.edu.hk/~cslzhang/paper/Supplement_LUT.pdf)

Skin color correction dataset s3 bucket
`s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/dehaze_dataset`

## Framework
<img src="figures/framework2.png" width="1024px"/>

## Usage
### Requirements
```bash 
conda create -y -n 3dlut_g5 python=3.8
conda activate 3dlut_g5 
pip install numpy==1.19.2 Pillow==6.1.0 opencv-python==3.4.8.29 scipy tqdm matplotlib 
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# torch 1.12 is required for onnx min() and max() conversion 
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
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
--lambda_monotonicity 0.0001 \
--lambda_smooth 10

# finetune with loss weight 
python image_adaptive_lut_train_paired.py \
--im_train ${data_dir}/train_im.txt \
--gt_train ${data_dir}/train_gt.txt \
--im_val ${data_dir}/val_im.txt \
--gt_val ${data_dir}/val_gt.txt \
--n_epochs 50 \
--output_dir LUTs/paired/${out} \
--lr 0.0001 \
--lambda_monotonicity 0.0001 \
--lambda_smooth 10 \
--epoch 39 \
--loss_weight 10
```

### Infer 
```bash
out=train_exp0
model_dir=./saved_models/LUTs/paired/${out}_sRGB
ep=39
ln -sf $model_dir/LUTs_${ep}.pth $model_dir/LUTs.pth
ln -sf $model_dir/classifier_${ep}.pth $model_dir/classifier.pth

python demo_eval_singleLUT.py --image_name $im --model_dir $model_dir --output_dir ./ --result_name_suffix "_${out}_ep${ep}"
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
