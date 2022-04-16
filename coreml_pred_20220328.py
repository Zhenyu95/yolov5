from coremltools.models import model
from utils.general import non_max_suppression, xyxy2xywh, xywhn2xyxy, xyxy2xywhn
import coremltools
import torch
import numpy as np
import random
import os
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import time
from tqdm import tqdm
import cv2
import math
import torch.nn as nn
from torchvision import transforms

# Parameters and Global variables
MODE = 'debug'
SAVE_IMG = True
VIEW_IMG = False
SAVE_TXT = True
SAVE_CONF = True
CAT_NAMES = ['Screw']

ML_EVAL = False

SW = 1  # SW = 1, pytorch export results
# SW = 2, Yanran's tf export results

# yolov5x6
# Anchor box can be checked in pytorch model
# ANCHORS = ([2.375,3.375, 5.5,5, 4.75,11.75],
#            [6,4.25, 5.375,9.5, 11.25,8.5625],
#            [4.375,9.40625, 9.46875,8.25, 7.43750,16.93750],
#            [6.81250,9.60938, 11.54688,5.93750, 14.45312,12.37500])
# # stide can be check in pytorch model
# stride = [8, 16, 32, 64]

# yolov5x
# Anchor box can be checked in pytorch model
ANCHORS = ([1.25000,  1.62500, 2.00000,  3.75000, 4.12500,  2.87500],
           [1.87500,  3.81250, 3.87500,  2.81250, 3.68750,  7.43750],
           [3.62500,  2.81250, 4.87500,  6.18750, 11.65625, 10.18750],
           )
# stide can be check in pytorch model
stride = [8, 16, 32]
# target size of input image (width, height)
IMG_SIZE = (960, 1280)
# confidence threshold
conf_thres = .35
area_thres = 300
edge_thres = 0.00  # roughly 80 pixel, image overlap ~10mm, 200 pixel,should be no impact

conf_color = .5  # conf_thres ~ conf_color show different color

if ML_EVAL:
    conf_color = .35
    SAVE_CONF = False
    edge_thres = 0.00

if SW == 1:
    kx = 4032/1280
    IMG_SIZE = (960, 1280)
elif SW == 2:
    kx = 4032/1280
    IMG_SIZE = (1280, 1280)

# Params and global variables that should be kept as it is
COLORS = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(CAT_NAMES))]
PATH = "./"
nc = len(CAT_NAMES)
nl = len(ANCHORS)
na = len(ANCHORS[0]) // 2
no = nc + 5
# initiate grid and anchor_grid
grid = [torch.zeros(1)] * nl
a = torch.tensor(ANCHORS).float().view(nl, -1, 2)
anchor_grid = [torch.zeros(1)] * nl

Template_path = '/Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/Template_Station1'

# Load Template Image
TFOV = []
for i in range(4):
    Temp = cv2.imread(Template_path + '/T'+str(i+1) +
                      '.jpg', cv2.IMREAD_GRAYSCALE)
    TFOV.append(Temp)

# Load FOV related template
Tsep = []
for i in range(4):
    T1 = cv2.imread(Template_path + '/FOV'+str(i+1) +
                    '/T1.jpg', cv2.IMREAD_GRAYSCALE)
    T01 = cv2.imread(Template_path + '/FOV'+str(i+1) +
                     '/T01.jpg', cv2.IMREAD_GRAYSCALE)
    T2 = cv2.imread(Template_path + '/FOV'+str(i+1) +
                    '/T2.jpg', cv2.IMREAD_GRAYSCALE)
    T02 = cv2.imread(Template_path + '/FOV'+str(i+1) +
                     '/T02.jpg', cv2.IMREAD_GRAYSCALE)
    Tsep.append([T1, T2, T01, T02])

# pattern match outputs
idx1 =  [(2893, 3387), (2862, 3326), (2981, 3376), (2817, 3427)]
idx2 =  [(2228, 2692), (351, 807), (648, 1033), (1904, 2472)]
idx3 =  [(1940, 2404), (1903, 2388), (2934, 3382), (2913, 3366)]
idx4 =  [(1009, 1468), (1527, 2000), (2438, 2926), (125, 629)]

n_point = 5
# load template

x1 = [394, 84, 182, 265]
y1 = [417, 437, 301, 270]
x2 = [142, 151, 110, 353]
y2 = [448, 436, 135, 120]

theta_s = [-math.pi/2, math.pi/2, -math.pi/2, math.pi/2]
flag = [1, 1, -1, -1]

# Set the paths
COREML_MODEL = ['/Users/zhenyu/Desktop/results/bundle-415ce438b60241dd9a991d9e1bcb0196/weights/best.mlmodel',
                ]
IMAGE_FOLDER = '/Users/zhenyu/Desktop/AOI7/Real/images'
OUT_FOLDER = IMAGE_FOLDER + '/Output'
os.makedirs(OUT_FOLDER, exist_ok=True)


def make_grid(nx=20, ny=20, i=0):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    anchor_grid = (a[i] * stride[i]).view((1, na, 1, 1, 2)
                                          ).expand(1, na, ny, nx, 2).float()
    return grid, anchor_grid


def resize_image(source_image):
    background = PIL.Image.new('RGB', IMG_SIZE, "black")
    source_image.thumbnail(IMG_SIZE)
    (w, h) = source_image.size
    background.paste(
        source_image, (int((IMG_SIZE[0] - w) / 2), int((IMG_SIZE[1] - h) / 2)))
    return background


def eval(image, model, file_name):
    resized = resize_image(image.copy())

    predictions = model.predict({'image': resized})

    z = []  # inference output
    x = []
    # for head in ['var_1763', 'var_1778', 'var_1793', 'var_1808']:
    #     x.append(torch.Tensor(predictions[head]))
    for head in ['var_1295', 'var_1308', 'var_1321']:
        x.append(torch.Tensor(predictions[head]))

    for i in range(nl):
        bs, _, ny, nx, _ = x[i].shape

        grid[i], anchor_grid = make_grid(nx, ny, i)

        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
        z.append(y.view(bs, -1, no))

    pred = torch.cat(z, 1)

    return pred


def eval1(image, model, file_name):
    img = image.copy()
    ori_height = img.size[1]
    ori_width = img.size[0]
    size = IMG_SIZE[0]

    trans = transforms.Compose([transforms.Resize((size, int(size*ori_width/ori_height))),
                                transforms.Pad((int((size-size*ori_width/ori_height)/2), 0)), ])

    img = np.array(trans(img))
    img = img[None]
    img = img/255.0

    prediction_result = model.predict({"input_1": img})
    prediction_boxes = torch.Tensor(prediction_result["Identity"])
    prediction_scores = torch.Tensor(prediction_result["Identity_1"])
    pred_t = torch.full_like(prediction_scores, 1)
    prediction_boxes = torch.squeeze(prediction_boxes)
    box = xyxy2xywh(prediction_boxes)
    box = box[None]
    pred = torch.cat([box, prediction_scores, pred_t], dim=2)

    return pred


def mean2(x):
    y = np.sum(x)/np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum()/math.sqrt((a*a).sum() * (b*b).sum())
    return r


def GetPredFOVID(image, T):
    f = nn.AvgPool2d(2, stride=2)
    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(), ])

    img0 = trans(image)
    img0 = img0[None]
    for i in range(4):
        img0 = f(img0)

    img0 = torch.squeeze(img0).numpy()
    coef1 = corr2(img0*255, T[0])
    coef2 = corr2(img0*255, T[1])
    coef3 = corr2(img0*255, T[2])
    coef4 = corr2(img0*255, T[3])
    score, idx = torch.max(torch.tensor([coef1, coef2, coef3, coef4]), 0)

    return idx


def GetMatchTemplate(img0, idxtt1, idxtt2, T):
    f = nn.AvgPool2d(2, stride=2)
    trans = transforms.Compose([
        transforms.ToTensor(), ])

    imgt1 = img0[idxtt1[0]-1:idxtt1[1], idxtt2[0]-1:idxtt2[1]]
    imgt01 = trans(imgt1)
    imgt01 = imgt01[None]
    # scaling image to speed up
    for i in range(2):
        imgt01 = f(imgt01)
    imgt01 = torch.squeeze(imgt01).numpy()*255
    imgt01 = imgt01.astype(np.uint8)

    T1 = T[0]
    T01 = T[1]
    # do template match roughly
    res = cv2.matchTemplate(imgt01, T01, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0]*4, max_loc[1]*4

    xs1 = k1-n_point if k1 > n_point else 0
    ys1 = h1-n_point if h1 > n_point else 0

    xe1 = k1+T1.shape[1]+n_point if k1+T1.shape[1] + \
        n_point < imgt1.shape[1] else imgt1.shape[1]
    ye1 = h1+T1.shape[0]+n_point if h1+T1.shape[0] + \
        n_point < imgt1.shape[0] else imgt1.shape[0]

    imgt1 = imgt1[ys1:ye1, xs1:xe1]
    # do pixel level template match
    res = cv2.matchTemplate(imgt1, T1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0], max_loc[1]

    return xs1+k1, ys1+h1


def GetROILine(image, TFOV, Tsep):
    # got image for FOV
    idx = GetPredFOVID(image, TFOV)
    # idx = 0
    # PIL image read cost time
    # get actual ROI x line & y line
    T1, T2, T01, T02 = Tsep[idx][0], Tsep[idx][1], Tsep[idx][2], Tsep[idx][3]
    idxt1, idxt2, idxt3, idxt4 = idx1[idx], idx2[idx], idx3[idx], idx4[idx]
    x01, y01, x02, y02 = x1[idx], y1[idx], x2[idx], y2[idx]
    theta_shift = theta_s[idx]

    # do template match
    img0 = np.array(image.convert('L'))
    k1, h1 = GetMatchTemplate(img0, idxt1, idxt2, [T1, T01])
    k2, h2 = GetMatchTemplate(img0, idxt3, idxt4, [T2, T02])

    #print(k1, h1, k2, h2)

    P1 = (k1+x01-1+idxt2[0]-1, h1+y01-1+idxt1[0]-1)
    P2 = (k2+x02-1+idxt4[0]-1, h2+y02-1+idxt3[0]-1)

    th1 = math.atan2(P2[1]-P1[1], P2[0]-P1[0])
    th2 = th1 + theta_shift

    if idx >= 2:
        th_temp = th1
        th1 = th2
        th2 = th_temp

    kt1 = math.tan(th1)
    kt2 = math.tan(th2)

    bt1 = P1[1] - kt1*P1[0]
    bt2 = P1[1]/kt2 - P1[0]

    return kt1, kt2, bt1, bt2, idx, img0


def pred(img_path, model_list):
    image = PIL.Image.open(os.path.join(IMAGE_FOLDER, img_path))
    # get ROI line separator
    kt1, kt2, bt1, bt2, idx, img0 = GetROILine(image, TFOV, Tsep)

    pred = torch.tensor([])
    for model in model_list:
        if SW == 1:
            pred = torch.cat((pred, eval(image, model, img_path)), 1)
        elif SW == 2:
            pred = torch.cat((pred, eval1(image, model, img_path)), 1)

    nms = non_max_suppression(pred, conf_thres, .6,
                              classes=None, agnostic=False)[0]
    label = []
    for *xyxy, conf, cls in nms:
        if SW == 2:
            xyxy[0] = xyxy[0] - (IMG_SIZE[0]-IMG_SIZE[0]
                                 * image.size[0]/image.size[1])/2
            xyxy[2] = xyxy[2] - (IMG_SIZE[0]-IMG_SIZE[0]
                                 * image.size[0]/image.size[1])/2
        xywh = xyxy2xywhn(torch.tensor(xyxy).view(
            1, 4), w=IMG_SIZE[0]*image.size[0]/image.size[1], h=IMG_SIZE[1]).view(-1).tolist()
        # if (xywh[2]*xywh[3]*4032*3024 > area_thres) and (xywh[0] > edge_thres) and (xywh[0] < 1-edge_thres) and (xywh[1] > edge_thres) and (xywh[1] < 1-edge_thres):
        if True:
            # add out of ROI filter function
            # if (xywh[0]*3024*math.sin(-theta_s[idx])*flag[idx] < (xywh[1]*4032/kt2 - bt2)*math.sin(-theta_s[idx])*flag[idx]) and (xywh[1]*4032 < kt1*xywh[0]*3024 + bt1):
                if SAVE_TXT:
                    label.append(('%g ' * 6 + '\n') % (cls, *xywh, conf))
                    print(label)
                    break
                if SAVE_IMG:
                    color = (255, 0, 0) if conf > conf_color else (0, 255, 255)
                    draw = PIL.ImageDraw.Draw(image)
                    draw.rectangle(np.array(torch.tensor(xyxy).view(
                        2, 2)*kx), outline=color, width=6)
                    # add class & conf
                    if SAVE_CONF:
                        c = int(cls)
                        label_txt = f'{CAT_NAMES[c]} {conf:.2f}'
                        font = PIL.ImageFont.truetype('Arial.ttf', max(
                            round(sum(IMG_SIZE) / 2 * 0.022), 12))
                        #font = PIL.ImageFont.truetype('Arial.ttf', 50)
                        w, h = font.getsize(label_txt)  # text width, height
                        outside = xyxy[1]*kx - h >= 0  # label fits outside box
                        draw.rectangle([xyxy[0]*kx,
                                        xyxy[1]*kx -
                                        h if outside else xyxy[1]*kx,
                                        xyxy[0]*kx + w + 1,
                                        xyxy[1]*kx + 1 if outside else xyxy[1]*kx + h + 1], fill=color)
                        draw.text((xyxy[0]*kx, xyxy[1]*kx - h if outside else xyxy[1]
                                  * kx), label_txt, fill=(255, 255, 255), font=font)

    if SAVE_TXT & (len(label) != 0):
        with open(os.path.join(OUT_FOLDER, '{}.txt'.format(img_path[:-4])), 'a') as f:
            for line in label:
                f.write(line)
        # with open(os.path.join(OUT_FOLDER, img_path.split('/')[0], 'classes.txt'), 'a') as f:
        #     f.write('screw\n')
    if SAVE_IMG:
        if not ML_EVAL:
            draw = PIL.ImageDraw.Draw(image)
            draw.line([(0, int(bt1)), (img0.shape[1], int(
                kt1*img0.shape[1]+bt1))], fill=(255, 0, 0), width=6)
            draw.line([(-int(bt2), 0), (int(img0.shape[0]/kt2-bt2),
                      img0.shape[0])], fill=(255, 0, 0), width=6)

        image.save(os.path.join(OUT_FOLDER, '{}.jpg'.format(img_path[:-4])))


def debug():
    model_list = []

    # Load the model
    for each in COREML_MODEL:
        print('Loading model {}...'.format(each))
        model_list.append(coremltools.models.model.MLModel(each))
        print('Model {} loaded'.format(each))

    time_tracker = {}
    time_sum = 0
    # list1 = os.listdir(IMAGE_FOLDER)
    # print(list1[158])
    for img_path in tqdm(os.listdir(IMAGE_FOLDER)):
        t0 = time.time()
        if (img_path.endswith(".jpg")) and (not img_path.startswith('.')):
            pred(img_path, model_list)
        delta_t = time.time() - t0
        time_tracker[img_path] = delta_t
    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum/len(time_tracker)))


def label_helper():

    model_list = []

    # Load the model
    for each in COREML_MODEL:
        model_list.append(coremltools.models.model.MLModel(each))
    time_tracker = {}
    time_sum = 0
    for img_folder in tqdm([each for each in os.listdir(IMAGE_FOLDER) if not each.startswith('.')]):
        for img_path in [each for each in os.listdir(os.path.join(IMAGE_FOLDER, img_folder)) if each.endswith('.jpg')]:
            if '{}.txt'.format(img_path[:-4]) in os.listdir(os.path.join(IMAGE_FOLDER, img_folder)):
                break
            t0 = time.time()
            img_path = os.path.join(img_folder, img_path)
            pred(img_path, model_list)
            delta_t = time.time() - t0
            time_tracker[img_path] = delta_t
    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum/len(time_tracker)))


def main():
    model_list = []

    # Load the model
    for each in COREML_MODEL:
        print('Loading model {}...'.format(each))
        model_list.append(coremltools.models.model.MLModel(each))
        print('Model {} loaded'.format(each))

    m = False
    while True:
        if not os.path.isfile(IMAGE_FOLDER+'1.jpg'):
            m = True

        if os.path.isfile(IMAGE_FOLDER+'1.jpg') & m:
            pred(IMAGE_FOLDER+'1.jpg', model_list
                 )
            m = False

        else:
            time.sleep(0.5)


if MODE == 'debug':
    debug()
else:
    main()
