import torch
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
import math
import torch.nn as nn
from torchvision import transforms
import cv2
import pandas as pd
import os
from yaml import Dumper, Loader, load, dump

def make_grid(nx, ny, i, a, stride, na):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    anchor_grid = (a[i] * stride[i]).view((1, na, 1, 1, 2)).expand(1, na, ny, nx, 2).float()
    return grid, anchor_grid


def resize_image(source_image, img_size):
    background = PIL.Image.new('RGB', img_size, "black")
    source_image.thumbnail(img_size)
    (w, h) = source_image.size
    background.paste(source_image, (int((img_size[0] - w) / 2), int((img_size[1] - h) / 2)))
    return background

def mean(x):
    y = np.sum(x) / np.size(x)
    return y


def corr(a, b):
    a = a - mean(a)
    b = b - mean(b)

    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


def load_template(template_path):
    # Load Template Image
    tfov = []
    for i in range(4):
        Temp = cv2.imread(template_path + '/T' + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
        tfov.append(Temp)

    # Load FOV related template
    tsep = []
    for i in range(4):
        T1 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T1.jpg', cv2.IMREAD_GRAYSCALE)
        # T01 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T01.jpg', cv2.IMREAD_GRAYSCALE)
        T2 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T2.jpg', cv2.IMREAD_GRAYSCALE)
        # T02 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T02.jpg', cv2.IMREAD_GRAYSCALE)
        tsep.append([T1, T2])
    return tfov, tsep


def GetPredFOVID(image, template):
    f = nn.AvgPool2d(16, stride=16)
    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(), ])

    img0 = trans(image)
    img0 = img0[None]
    img0 = f(img0)

    img0 = torch.squeeze(img0).numpy()
    coef1 = corr(img0 * 255, template[0])
    coef2 = corr(img0 * 255, template[1])
    coef3 = corr(img0 * 255, template[2])
    coef4 = corr(img0 * 255, template[3])
    score, idx = torch.max(torch.tensor([coef1, coef2, coef3, coef4]), 0)

    return idx


def GetMatchTemplate(img0, idxtt1, idxtt2, template, n_point=5):
    f = nn.AvgPool2d(4, stride=4)
    trans = transforms.Compose([
                transforms.ToTensor(),])
    
    # crop the image to get the search region
    imgt1  = img0[idxtt1[0]:idxtt1[1], idxtt2[0]:idxtt2[1]]
    imgt01 = trans(imgt1)
    imgt01 = imgt01[None]
    ## scaling image to speed up
    imgt01 = f(imgt01)
    imgt01 = torch.squeeze(imgt01).numpy()*255
    imgt01 = imgt01.astype(np.uint8)

    t1  = template
    t01 = torch.squeeze(f(trans(t1)))*255
    t01 = np.array(t01, dtype=np.uint8)
    ## do template match roughly
    res = cv2.matchTemplate(imgt01, t01, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0]*4, max_loc[1]*4

    xs1 = k1-n_point if k1 > n_point else 0
    ys1 = h1-n_point if h1 > n_point else 0

    xe1 = k1+t1.shape[1]+n_point if k1+t1.shape[1]+n_point < imgt1.shape[1] else imgt1.shape[1]
    ye1 = h1+t1.shape[0]+n_point if h1+t1.shape[0]+n_point < imgt1.shape[0] else imgt1.shape[0]

    imgt1 = imgt1[ys1:ye1, xs1:xe1]
    ## do pixel level template match
    res = cv2.matchTemplate(imgt1, t1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0], max_loc[1]

    return xs1+k1, ys1+h1


def GetROILine(image, tfov, tsep, idxs, xy, theta_s):
    # got image for FOV
    idx = GetPredFOVID(image, tfov)
    # PIL image read cost time
    # get actual ROI x line & y line
    t1, t2 = tsep[idx]
    idxt1, idxt2, idxt3, idxt4 = idxs[:, 3]
    x01, y01, x02, y02 = xy[idx, :]
    theta_shift = theta_s[idx]

    # do template match
    # convert the image to grayscale
    img0 = np.array(image.convert('L'))
    k1, h1 = GetMatchTemplate(img0, idxt1, idxt2, t1)
    k2, h2 = GetMatchTemplate(img0, idxt3, idxt4, t2)

    #print(k1, h1, k2, h2)

    P1 = (k1 + x01 - 1 + idxt2[0] - 1, h1 + y01 - 1 + idxt1[0] - 1)
    P2 = (k2 + x02 - 1 + idxt4[0] - 1, h2 + y02 - 1 + idxt3[0] - 1)

    th1 = math.atan2(P2[1] - P1[1], P2[0] - P1[0])
    th2 = th1 + theta_shift

    if idx >= 2:
        th_temp = th1
        th1 = th2
        th2 = th_temp

    kt1 = math.tan(th1)
    kt2 = math.tan(th2)

    bt1 = P1[1] - kt1 * P1[0]
    bt2 = P1[1] / kt2 - P1[0]

    return kt1, kt2, bt1, bt2, idx, img0


def Tensor2Image(img1):
    
    img1 = torch.squeeze(img1).numpy()
    img1 = img1.transpose((1,2,0))
    img1 = np.array(img1*255, dtype = np.uint8)
    img2 = PIL.Image.fromarray(img1)

    return img2


# code copied from our beloved Steve's GenerateTemplate.py
def generate_template(file_path, fov_list):

    f1 = nn.AvgPool2d(2, stride=2)
    f2 = nn.AvgPool2d(16, stride=16)
    trans = transforms.Compose([transforms.ToTensor(),])

    idx1, idx2, idx3, idx4 = [], [], [], []
    xy = []
    
    for i, fn in enumerate(fov_list):
        img0 = PIL.Image.open(file_path + '/' + fn + '.jpg')
        img0 = trans(img0)
        img0 = img0[None]

        imgt = f2(img0)
        imgt = Tensor2Image(imgt)
        imgt.save(file_path + '/T'+str(i+1)+'.jpg', quality=100)
        
        _, _, m, n = img0.shape
        Gain = [1, n, m, n, m]

        #### read txt
        df_bw = pd.read_csv(file_path + '/' + fn + '.txt', sep=' ', header = None, names=['class', 'x', 'y', 'w', 'h'], index_col=False)
        df_1  = df_bw[df_bw['class'].isin([0])].reset_index(drop = True)
        df_2  = df_bw[df_bw['class'].isin([1])].reset_index(drop = True)

        df_1 = df_1 * Gain
        df_2 = df_2 * Gain
        # print(df_1)
        os.makedirs(file_path + '/' + fn, exist_ok=True)

        for j in range(len(df_1)):
            imgt0 = img0[:, :, int(df_1.loc[j, 'y']- df_1.loc[j, 'h']/2):int(df_1.loc[j, 'y']+ df_1.loc[j, 'h']/2)+1,\
                int(df_1.loc[j, 'x']- df_1.loc[j, 'w']/2):int(df_1.loc[j, 'x']+ df_1.loc[j, 'w']/2)+1]
            
            imgt1 = Tensor2Image(imgt0)
            imgt1.save(file_path + '/' + fn +'/T'+str(j+1)+'.jpg', quality=100)

            imgt2 = f1(imgt0)
            imgt3 = Tensor2Image(imgt2)
            imgt3.save(file_path + '/' + fn +'/T0'+str(j+1)+'.jpg', quality=100)

            if j == 0:
                idx1.append((int(df_2.loc[j, 'y']- df_2.loc[j, 'h']/2), int(df_2.loc[j, 'y']+ df_2.loc[j, 'h']/2)+1))
                idx2.append((int(df_2.loc[j, 'x']- df_2.loc[j, 'w']/2), int(df_2.loc[j, 'x']+ df_2.loc[j, 'w']/2)+1))
            else:
                idx3.append((int(df_2.loc[j, 'y']- df_2.loc[j, 'h']/2), int(df_2.loc[j, 'y']+ df_2.loc[j, 'h']/2)+1))
                idx4.append((int(df_2.loc[j, 'x']- df_2.loc[j, 'w']/2), int(df_2.loc[j, 'x']+ df_2.loc[j, 'w']/2)+1))
            xy.append([imgt1.width, imgt1.height])
    mask = np.array([[1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0]])
    offset = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    xy = (np.array(xy).reshape((4,4)))*mask + offset
    idxs = np.array([idx1, idx2, idx3, idx4])
    return idxs, xy


# to get idx1, idx2, idx3, idx4 and pack into a tuple idxs
def get_idxs(new=False, file_path=None, fov_list=['FOV1', 'FOV2', 'FOV3', 'FOV4']):
    if new:
        idxs, xy = generate_template(file_path, fov_list)
        
        with open('template_idxs.yaml', 'w') as f:
            dump((idxs, xy), f)
    else:
        with open('template_idxs.yaml') as f:
            idxs, xy = load(f, Loader=Loader)
    return idxs, xy
