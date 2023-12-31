# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import cv2
from skimage import io
import numpy as np
import detect_word.utils.craft_utils as craft_utils # some convert function built-in process of CRAFT model
import detect_word.utils.imgproc as imgproc# image procesing - normalize, denormalize v.v
import detect_word.utils.file_utils as file_utils# working with folder, save, load dir etc

from detect_word.utils.craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold = 0.75, link_threshold=0.5, low_text=0.3, cuda = 0, poly = False, refine_net=None):
    
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    # image size is resize base on min of mag_ratio * max(w,h) or square
    # moreover, image is padded zeros for enough some number % 32 ==0 , ratio is based on this
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized) # normal follow std, mean of Imagenet, which consistent with pretrain VGG
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
        net = net.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x) # tu truoc den day het tam 1.05s | output la score ve text va link, can postprocessing de su dung

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy() #minh cai nay het 0.3s 
    score_link = y[0,:,:,1].cpu().data.numpy() #minh cai nay het 0.3s 

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly) #1.7s

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h) #1.8s
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

import time
def inferDetect(img_path,cuda=1):
    image_list, _, _ = file_utils.get_files(img_path)
    # load net
    net = CRAFT()     # initialize
    path_model = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/weights/craft_mlt_25k.pth'
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(path_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(path_model, map_location='cpu')))
    net.eval()
    t = time.time()
    # load data
    for k, image_path in enumerate(image_list):
        # print("Test image {}/{}: {}".format(k+1,len(image_list),image_path))
        image = imgproc.loadImage(image_path) # chi don gian la doc cai anh theo RGB thoi | 0.006s
        polys= test_net(net, image,cuda)

        # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder,draw=True) # see file_utils.py to set output is image, txt or ste

    print(f'Done detect word after {round(time.time() - t,6)}s ')
    return polys

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    # /home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/weights/craft_mlt_25k.pth
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default=os.path.join(here,'weights/craft_mlt_25k.pth'), type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.75, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.5, type=float, help='link confidence threshold - bigger is predict with character')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show-time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test-path', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default=os.path.join(here,'weights/craft_refiner_CTW1500.pth'), type=str, help='pretrained refiner model')

    args = parser.parse_args()


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_path)

    result_folder = os.path.join(here,'result/')
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
        # load net
    net = CRAFT()     # initialize

    # print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    net.eval()
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    # LinkRefiner
    refine_net = None
    if args.refine:
        from utils.refinenet import RefineNet
        refine_net = RefineNet()
        # print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
    # load data
    for k, image_path in enumerate(image_list):
        # print("Test image {}/{}: {}".format(k+1,len(image_list),image_path))
        image = imgproc.loadImage(image_path) # chi don gian la doc cai anh theo RGB thoi 
        bboxes, polys = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder,draw=True) # see file_utils.py to set output is image, txt or ste
    if args.refine:
        is_refine = True
    else:
        is_refine = False
    print("Done ! Time processed : {}s , use_refine = {}".format(round(time.time() - t,6),is_refine))