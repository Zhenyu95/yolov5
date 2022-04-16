from utils.general import non_max_suppression, xyxy2xywh, xyxy2xywhn
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
from pred_utils import *

# Parameters and Global variables
# MODE = 'debug' -> debuging loacally
# MODE = 'main'  -> running mode on the production line
MODE = 'debug'
# if images with predicted labels need to be saved
SAVE_IMG = True
# if images with predicted labels need displayed once prediction finished
VIEW_IMG = False
# if predicted labels need to be saved in YOLO format
SAVE_TXT = True
# if need to save an extra column of confidence level
# as the last column of predicted labels
SAVE_CONF = True
# names of each category
CAT_NAMES = ['Screw']

# MODEL_SOURCE = 'pytorch' or 'tf' 
# 'pytorch': the coreml model is converted from YOLOv5 model
# 'tf': the model is converted from BJML's tensorflow model
MODEL_SOURCE = 'pytorch'

# Anchor box can be checked in pytorch model only (coreml model does not work)
# ANCHORS of yolov5x
ANCHORS = ([1.25000, 1.62500, 2.00000, 3.75000, 4.12500, 2.87500],
           [1.87500, 3.81250, 3.87500, 2.81250, 3.68750, 7.43750],
           [3.62500, 2.81250, 4.87500, 6.18750, 11.65625, 10.18750],
           )
# ANCHORS of yolov5x6
# ANCHORS = ([2.37500, 3.37500, 5.50000, 5.00000, 4.75000, 11.75000],
#            [6.00000, 4.25000, 5.37500, 9.50000, 11.25000, 8.56250],
#            [4.37500, 9.40625, 9.46875, 8.25000, 7.43750, 16.93750],
#            [6.81250, 9.60938, 11.54688, 5.93750, 14.45312, 12.37500])
# STRIDE can be checked in pytorch model only (coreml model does not work)
# STRIDE of yolov5x
STRIDE = [8, 16, 32]
# STRIDE of yolov5x6
# STRIDE = [8, 16, 32, 64]

# target size of input image (width, height)
IMG_SIZE = (960, 1280)
RAW_IMG_SIZE = (3024, 4032)
kx = RAW_IMG_SIZE[1]/IMG_SIZE[1]
# confidence threshold used in NMS
CONF_THRESH = .35
AREA_THRESH = 300
EDGE_THRESH = 0.00  # roughly 80 pixel, image overlap ~10mm, 200 pixel,should be no impact
# if confidence > CONF_COLOR_THRESH use colorA else colorB
CONF_COLOR_THRESH = .5 


# Params and global variables that should be kept as it is
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CAT_NAMES))]
nc = len(CAT_NAMES)
nl = len(ANCHORS)
na = len(ANCHORS[0]) // 2
no = nc + 5
# initiate grid and anchor_grid
grid = [torch.zeros(1)] * nl
anchor = torch.tensor(ANCHORS).float().view(nl, -1, 2)
anchor_grid = [torch.zeros(1)] * nl

TEMPLATE_PATH = './Template'


flag = [1, 1, -1, -1]
THETA_S = [-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2]

# Load Complete

COREML_MODEL = ['/Users/zhenyu/Desktop/results/bundle-415ce438b60241dd9a991d9e1bcb0196/weights/best.mlmodel',
                ]
IMAGE_FOLDER = '/Users/zhenyu/Desktop/AOI7/Real/images'
OUT_FOLDER = IMAGE_FOLDER + '/Output'
os.makedirs(OUT_FOLDER, exist_ok=True)

def eval(image, model, mode=MODEL_SOURCE, stride=STRIDE, na=na):
    # if model is converted from BJML team's tensorflow model, 
    # predict the result via the following code
    if mode == 'tf':
        img = image.copy()
        ori_height = img.size[1]
        ori_width = img.size[0]
        size = IMG_SIZE[0]

        trans = transforms.Compose([transforms.Resize((size, int(size * ori_width / ori_height))),
                                    transforms.Pad((int((size - size * ori_width / ori_height) / 2), 0)), ])

        img = np.array(trans(img))
        img = img[None]
        img = img / 255.0

        prediction_result = model.predict({"input_1": img})
        prediction_boxes = torch.Tensor(prediction_result["Identity"])
        prediction_scores = torch.Tensor(prediction_result["Identity_1"])
        pred_t = torch.full_like(prediction_scores, 1)
        prediction_boxes = torch.squeeze(prediction_boxes)
        box = xyxy2xywh(prediction_boxes)
        box = box[None]
        pred = torch.cat([box, prediction_scores, pred_t], dim=2)

        return pred
    # if mode is converted from pytorch YOLOv5,
    # predict the result via the following code
    else:
        resized = resize_image(image.copy(), IMG_SIZE)

        predictions = model.predict({'image': resized})

        z = []  # inference output
        x = []
        
        for head in model.output_description:
            x.append(torch.Tensor(predictions[head]))

        for i in range(nl):
            bs, _, ny, nx, _ = x[i].shape

            grid[i], anchor_grid = make_grid(nx, ny, i, anchor, stride, na)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * STRIDE[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            z.append(y.view(bs, -1, no))

        pred = torch.cat(z, 1)

        return pred

def pred(img_path, model_list, image_folder, tfov, tsep, idxs, xy, mode=MODEL_SOURCE, theta_s=THETA_S):
    image = PIL.Image.open(os.path.join(image_folder, img_path))
    # get ROI line separator
    kt1, kt2, bt1, bt2, idx, img0 = GetROILine(image, tfov, tsep, idxs, xy, theta_s)

    pred = torch.tensor([])
    for model in model_list:
        pred = torch.cat((pred, eval(image, model, mode)), 1)

    nms = non_max_suppression(pred, CONF_THRESH, .6, classes=None, agnostic=False)[0]
    label = []
    for *xyxy, conf, cls in nms:
        # if MODEL_SOURCE == 'tf', remove black area
        if MODEL_SOURCE == 'tf':
            xyxy[0] = xyxy[0] - (IMG_SIZE[0] - IMG_SIZE[0] * image.size[0] / image.size[1]) / 2
            xyxy[2] = xyxy[2] - (IMG_SIZE[0] - IMG_SIZE[0] * image.size[0] / image.size[1]) / 2
            xywh = xyxy2xywhn(torch.tensor(xyxy).view(
                1, 4), w=IMG_SIZE[0] * image.size[0] / image.size[1], h=IMG_SIZE[1]).view(-1).tolist()
        xywh = xyxy2xywhn(torch.tensor(xyxy).view(1, 4), 
                          w=IMG_SIZE[0], 
                          h=IMG_SIZE[1]
                          ).view(-1).tolist()
        if (xywh[2] * xywh[3] * 4032 * 3024 > AREA_THRESH) and (xywh[0] > EDGE_THRESH) and (xywh[0] < 1 - EDGE_THRESH) and (xywh[1] > EDGE_THRESH) and (xywh[1] < 1 - EDGE_THRESH):
            # add out of ROI filter function
            if (xywh[0] * 3024 * math.sin(-theta_s[idx]) * flag[idx] < (xywh[1] * 4032 / kt2 - bt2) * math.sin(-theta_s[idx]) * flag[idx]) and (xywh[1] * 4032 < kt1 * xywh[0] * 3024 + bt1):
                if SAVE_TXT:
                    label.append(('%g ' * 6 + '\n') % (cls, *xywh, conf))
                if SAVE_IMG:
                    color = (255, 0, 0) if conf > CONF_COLOR_THRESH else (0, 255, 255)
                    draw = PIL.ImageDraw.Draw(image)
                    draw.rectangle(np.array(torch.tensor(xyxy).view(2, 2) * kx), outline=color, width=6)
                    # add class & conf
                    if SAVE_CONF:
                        c = int(cls)
                        label_txt = f'{CAT_NAMES[c]} {conf:.2f}'
                        font = PIL.ImageFont.truetype('Arial.ttf', max(round(sum(IMG_SIZE) / 2 * 0.022), 12))
                        #font = PIL.ImageFont.truetype('Arial.ttf', 50)
                        w, h = font.getsize(label_txt)  # text width, height
                        outside = xyxy[1] * kx - h >= 0  # label fits outside box
                        draw.rectangle([xyxy[0] * kx,
                                        xyxy[1] * kx - h if outside else xyxy[1] * kx,
                                        xyxy[0] * kx + w + 1,
                                        xyxy[1] * kx + 1 if outside else xyxy[1] * kx + h + 1], fill=color)
                        draw.text((xyxy[0] * kx, xyxy[1] * kx - h if outside else xyxy[1] * kx),
                                  label_txt, fill=(255, 255, 255), font=font)

    if SAVE_TXT & (len(label) != 0):
        with open(os.path.join(OUT_FOLDER, '{}.txt'.format(img_path[:-4])), 'a') as f:
            for line in label:
                f.write(line)
    if SAVE_IMG & (len(label) != 0):
        draw = PIL.ImageDraw.Draw(image)
        draw.line([(0, int(bt1)), (img0.shape[1], int(kt1 * img0.shape[1] + bt1))], fill=(255, 0, 0), width=6)
        draw.line([(-int(bt2), 0), (int(img0.shape[0] / kt2 - bt2), img0.shape[0])], fill=(255, 0, 0), width=6)

        image.save(os.path.join(OUT_FOLDER, '{}.jpg'.format(img_path[:-4])))


def debug(image_folder=IMAGE_FOLDER, template_path=TEMPLATE_PATH, ):
    model_list = []

    # Load the model
    for each in COREML_MODEL:
        print('Loading model {}...'.format(each))
        model_list.append(coremltools.models.model.MLModel(each))
        print('Model {} loaded'.format(each))
        
    tfov, tsep = load_template(template_path)
    idxs, xy = get_idxs(new=True, file_path=template_path)

    time_tracker = {}
    time_sum = 0
    for img_path in tqdm(os.listdir(image_folder)):
        t0 = time.time()
        if (img_path.endswith(".jpg")) and (not img_path.startswith('.')):
            pred(img_path, model_list, image_folder, tfov, tsep, idxs, xy)
        delta_t = time.time() - t0
        time_tracker[img_path] = delta_t
    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum / len(time_tracker)))


def main(image_folder=IMAGE_FOLDER, template_path=TEMPLATE_PATH):
    model_list = []

    # Load the model
    for each in COREML_MODEL:
        print('Loading model {}...'.format(each))
        model_list.append(coremltools.models.model.MLModel(each))
        print('Model {} loaded'.format(each))
    
    tfov, tsep = load_template(template_path)

    m = False
    while True:
        if not os.path.isfile(image_folder + '1.jpg'):
            m = True

        if os.path.isfile(image_folder + '1.jpg') & m:
            pred(image_folder + '1.jpg', model_list, image_folder, tfov, tsep)
            m = False

        else:
            time.sleep(0.5)

if MODE == 'debug':
    debug()
else:
    main()

