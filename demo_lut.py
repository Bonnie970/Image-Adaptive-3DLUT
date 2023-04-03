import argparse
import torch
import os
import numpy as np
import cv2
from PIL import Image
import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob 

# from models import *
from models_x import *
import torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser()

# frames = ['demo_images/sRGB/1540019057_1010_00084444.png', 'demo_images/sRGB/1540019057_1074_00089641.png', 'demo_images/sRGB/1540019057_1081_00090506.png']
# frames = ['demo_images/sRGB/hazy6.png','demo_images/sRGB/hazy7.png']
frames = ['demo_images/sRGB/hazy4_dark.png', 'demo_images/sRGB/hazy8_dark.png']
print(frames)
output_dir = 'lut_study'
model_dir = 'pretrained_models/sRGB'

os.makedirs(output_dir, exist_ok=True)

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
#LUT3 = Generator3DLUT_zero()
#LUT4 = Generator3DLUT_zero()
classifier = Classifier()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    #LUT3 = LUT3.cuda()
    #LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("%s/LUTs.pth" % model_dir)
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
#LUT3.load_state_dict(LUTs["3"])
#LUT4.load_state_dict(LUTs["4"])
LUT0.eval()
LUT1.eval()
LUT2.eval()
#LUT3.eval()
#LUT4.eval()
classifier.load_state_dict(torch.load("%s/classifier.pth" % model_dir))
classifier.eval()


def generate_LUT(img):

    pred = classifier(img).squeeze()
    
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    return LUT, pred.detach().to('cpu').numpy()

for image_path in frames:
    # img = Image.open(image_path)
    img = cv2.imread(image_path)[:,:,::-1]
    print(img.shape, img.min(), img.max())

    img = TF.to_tensor(img.copy()).type(Tensor)
    img = img.unsqueeze(0)
    # infer 
    LUT, pred = generate_LUT(img)

    results = []
    luts = [LUT0.LUT, LUT1.LUT, LUT2.LUT]
    lut_downsample = [1,1,1] #[0.7, 0.8, 0.9]
    for i in range(3):
        test_weights = pred.copy()
        test_weights[0] = test_weights[0] * lut_downsample[i] # decrease the impact of i-th LUT 
        lut = test_weights[0] * luts[0] + test_weights[1] * luts[1] + test_weights[2] * luts[2]
        _, result = trilinear_(lut, img)
        print(result.shape, test_weights)
        results.append(result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    # geenrate normal output  
    lut = pred[0] * luts[0] + pred[1] * luts[1] + pred[2] * luts[2] 
    _, result = trilinear_(lut, img)
    results.append(result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    # plot 
    im1 = cv2.imread(image_path)[:,:,::-1]
    
    # apply weights 
    for i in range(len(results)):
        results[i] = results[i] * (1 - 0.25 * i) + im1 * (0.25 * i)

    im2, im3, im4, im5 = results 

    # put text 
    cv2.putText(im1, 'SDR', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
    cv2.putText(im2, 'LUT_0 0.75', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
    cv2.putText(im3, 'LUT_1 0.75', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
    cv2.putText(im4, 'LUT_2 0.75', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)

    # print(im1.shape,im2.shape,im3.shape,im4.shape )

    im = np.vstack((np.hstack((im1, im2, im3)), np.hstack((im4, im5, np.zeros(im1.shape)))))

    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), im[:, :, ::-1])



