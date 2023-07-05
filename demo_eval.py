import argparse
import torch
import os
import numpy as np
import cv2
from PIL import Image
import glob 
from tqdm import tqdm
# import matplotlib.pyplot as plt

# from models import *
from models_x import *
import torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF

import time 


parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", type=str, default="demo_images", help="directory of image")
parser.add_argument("--image_name", type=str, default="demo_images/sRGB/a1629.jpg", help="name of image")
parser.add_argument("--result_name_suffix", type=str, default="", help="suffix to add to image result")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--video", type=int, default=0, help="generate video or not")
parser.add_argument("--model_dir", type=str, default="pretrained_models/sRGB", help="directory of pretrained models")
parser.add_argument("--output_dir", type=str, default="demo_results", help="directory to save results")
opt = parser.parse_args()
# opt.model_dir = opt.model_dir + '/' + opt.input_color_space
if '*' not in opt.image_name:
    opt.image_path = opt.image_name # opt.image_dir + '/' + opt.input_color_space + '/' + opt.image_name
else:
    opt.image_path = opt.image_name
os.makedirs(opt.output_dir, exist_ok=True)

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
LUTs = torch.load("%s/LUTs.pth" % opt.model_dir)
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
classifier.load_state_dict(torch.load("%s/classifier.pth" % opt.model_dir))
classifier.eval()


def generate_LUT(img, fix_pred=None):
    if fix_pred is not None:
        pred = fix_pred
    else:
        pred = classifier(img).squeeze()
    
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT #+ pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    return LUT, pred.detach().to('cpu').numpy()


def RGB2sRGB(image, gamma=2.2):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def sRGB2RGB(image, gamma=2.2):
	table = np.array([((i / 255.0) ** gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# ----------
#  test
# ----------
max_time = 0
times = []
if not opt.video:
    combine_ratio = 1
    if os.path.isdir(opt.image_path):
        frames = sorted(glob.glob(os.path.join(opt.image_path, '*')))
    elif opt.image_path.endswith('.txt'):
        frames = [x for x in open(opt.image_path, 'r').read().split('\n') if x!='']
    else:
        frames = [opt.image_path]

    preds = []
    fix_pred = torch.tensor([1.129625 , -0.5558822, -0.6274552]).cuda() #torch.tensor([ 1.1643001 , -0.42816073, -0.4653956 ]).cuda() # None # torch.tensor([ 1.1667712, -1.1818283, -0.7892431]).cuda()
    for frame in tqdm(frames):
        # frame = '/home/ubuntu/dehaze_e2e/results/Ranji-Output_New_15000/00000001_dcp.png'
        # print(frame)
        img = Image.open(frame)
        # print(np.asarray(img), np.asarray(img).shape)
        img = TF.to_tensor(img).type(Tensor)
        img = img.unsqueeze(0)

        #print(img.permute(0,2,3,1), img.dtype, img.shape)
        #assert 0==1
        
        # model run 
        t1 = time.time()
        LUT, pred = generate_LUT(img, fix_pred=fix_pred)
        preds.append(pred)
        
        
        result = trilinear_(LUT, img)

        t2 = time.time()
        times.append(t2-t1)

        if (t2-t1) > max_time:
            print('max time @', t2-t1, 'image ', frame)
            max_time = t2-t1
        
        # save result
        output = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # output = combine_ratio * output + (1-combine_ratio) * img
        output = output.astype(np.uint8)[:, :, ::-1]
        output = np.hstack([cv2.imread(frame), output])
        cv2.imwrite(os.path.join(opt.output_dir, os.path.basename(frame).replace('.', opt.result_name_suffix + '.')), output)

print(np.mean(times)*1000, np.max(times)*1000, np.min(times)*1000)
np.save('{}.npy'.format('_'.join(opt.image_path.split('/')[-2:])), preds)


# geenrate video 
if opt.video:
    combine_ratio = [0.5, 0.75]
    if '*' in opt.image_path:
        frames = sorted(glob.glob(opt.image_path))
        video_name = os.path.join(opt.output_dir, f'result_xyz.mp4')
        _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        im1 = cv2.imread(frames[0])
        print(im1.shape)
        _out = cv2.VideoWriter(video_name, _fourcc, 20.0, (im1.shape[1]*2, im1.shape[0]*2))
        for frame in frames:
            im1 = cv2.imread(frame) 
            im2 = cv2.imread(os.path.join(opt.output_dir, os.path.basename(frame).replace('.', f'_1.')))
            im3 = im1 * (1 - combine_ratio[0]) + im2 * combine_ratio[0]
            im4 = im1 * (1 - combine_ratio[1]) + im2 * combine_ratio[1]
            # put text 
            im1 = cv2.putText(im1, 'SDR', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
            im2 = cv2.putText(im2, 'Enhanced SDR 1.0', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
            im3 = cv2.putText(im3, 'Enhanced SDR 0.5', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
            im4 = cv2.putText(im4, 'Enhanced SDR 0.75', (25, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
            im = np.vstack((np.hstack((im1, im2)), np.hstack((im3, im4))))
            _out.write(im.astype(np.uint8))
        _out.release()





    # if '*' not in opt.image_path:
    #     # read image and transform to tensor
    #     if opt.input_color_space == 'sRGB':
    #         # img = Image.open(opt.image_path)
    #         # img = TF.to_tensor(img).type(Tensor)

    #         # rgb to sRGB 
    #         img = cv2.imread(opt.image_path)[:,:,::-1]
    #         img = RGB2sRGB(img, gamma=2.2)
    #         img = TF_x.to_tensor(img).type(Tensor)
            
    #     elif opt.input_color_space == 'XYZ':
    #         img = cv2.imread(opt.image_path, -1)

    #         # img = cv2.imread('./data_cwc/frames/1540019057_1010_00084377.png', -1)
    #         # img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    #         # img = np.array(img/255*65535, np.uint16)

    #         img = np.array(img)
    #         print(img.shape, img.max(), img.min(), img.dtype)
    #         img = TF_x.to_tensor(img).type(Tensor)
    #     img = img.unsqueeze(0)

    #     LUT, pred = generate_LUT(img)

    #     # generate image
    #     # result = trilinear_(LUT, img)
    #     result = trilinear_(LUT, img)

    #     # save image
    #     output = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #     # img = np.array(Image.open(opt.image_path))
    #     # for combine_ratio in [0, 0.25, 0.5, 0.75, 1]:
    #     #     _output = combine_ratio * output + (1-combine_ratio) * img
    #     #     _output = _output.astype(np.uint8)
    #     #     cv2.imwrite(f'{opt.output_dir}/result_{combine_ratio}.png', _output[:, :, ::-1])
    #     output = sRGB2RGB(output, 2.2)
    #     output = Image.fromarray(output)
    #     output.save('%s/result_sRGB.png' % opt.output_dir, quality=95)
    # else:
