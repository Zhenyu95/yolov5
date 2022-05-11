from utils.general import non_max_suppression, xyxy2xywh, xyxy2xywhn
import coremltools
import torch
import numpy as np
import os
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import time
from tqdm import tqdm
import math
from torchvision import transforms
from pred_utils import *
import cv2

""" Global variables : Path variables are defined below """

# COREML_MODEL is a list containing all coreml model path
COREML_MODEL = ['/Users/zhenyu/Downloads/export.mlmodel']
IMAGE_FOLDER = '/Users/zhenyu/Desktop/AOI7'
TEMPLATE_PATH = '/Users/zhenyu/Desktop/AOI7/Template'

# folder structure should follow the below if evaluate()
# - folder
#   - FIT
#       - images
#       - labels
#   - OK_Val
#       - images

""" Global variables : Image size are defined below """

# target size of input image (width, height)
IMG_SIZE = (960, 1280)
RAW_IMG_SIZE = (3024, 4032)


""" Global variables : General settings are defined below """

# MODE = 'debug' -> debuging loacally
# MODE = 'main'  -> running mode on the production line
# MODE = 'eval'  -> evaluating a model
MODE = 'eval'


""" Global variables : Parameters of model are defined below """

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

# confidence threshold used in NMS
CONF_THRESH = .35
# iou threshold used in NMS
IOU_THRESH  = 0.45
AREA_THRESH = 300
EDGE_THRESH = 0.00  # roughly 80 pixel, image overlap ~10mm, 200 pixel,should be no impact
# if confidence > CONF_COLOR_THRESH use colorA else colorB
CONF_COLOR_THRESH = .5 


""" Global variables : Debug() and Main() predicting settings """

# if images with predicted labels need to be saved
SAVE_IMG = True
# if images with predicted labels need displayed once prediction finished
VIEW_IMG = False
# if predicted labels need to be saved in YOLO format
SAVE_TXT = True
# if need to draw confidence level in the result image
SAVE_CONF = True
# names of each category
CAT_NAMES = ['Screw']
# check rubber
rubbercheck=True



""" Global variables : below variables should be kept as it is """

nc = len(CAT_NAMES)
nl = len(ANCHORS)
na = len(ANCHORS[0]) // 2
no = nc + 5
# initiate grid and anchor_grid
grid = [torch.zeros(1)] * nl
anchor = torch.tensor(ANCHORS).float().view(nl, -1, 2)
anchor_grid = [torch.zeros(1)] * nl
kx = RAW_IMG_SIZE[1]/IMG_SIZE[1]


# detect() decodes results from model without nms
def detect(image, model, source, stride=STRIDE, na=na):
    # if model is converted from BJML team's tensorflow model, 
    # predict the result via the following code
    if source == 'tf':
        img = resize_image(image, IMG_SIZE, source)

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
        img = resize_image(image, IMG_SIZE, source)

        predictions = model.predict({'image': img})

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

def predict(img_path, model_list, image_folder, template, source,
         conf_thresh = CONF_THRESH,
         edge_thresh = EDGE_THRESH,
         area_thresh = AREA_THRESH,
         iou_thresh = IOU_THRESH,
         save_text = SAVE_TXT,
         save_conf = SAVE_CONF,
         save_img = SAVE_IMG
         ):
    
    image = PIL.Image.open(os.path.join(image_folder, img_path))
    out_folder = os.path.join(image_folder, '../output')
    
    #unpack tfov, tseq, idxs, xy from template
    tfov, tsep, idxs, xy = template
    # get image for FOVid
    idx = GetPredFOVID(image, tfov)
    
    theta_s = [-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2]
    flag = [1, 1, -1, -1]

    # get ROI line separator
    kt1, kt2, bt1, bt2, img0, P1 = GetROILine(image, idx, tsep, idxs, xy, theta_s)

    pred = torch.tensor([])
    for model in model_list:
        pred = torch.cat((pred, detect(image, model, source)), 1)

    nms = non_max_suppression(pred, conf_thresh, .45, classes=None, agnostic=False)[0]
    
    if rubbercheck:
        nms = checkrubber(nms, image, idx, P1)

    label_list = []
    for *xyxy, conf, cls in nms:
        # if MODEL_SOURCE == 'tf', remove black area
        if source == 'tf':
            xyxy[0] = xyxy[0] - (IMG_SIZE[0] - IMG_SIZE[0] * image.size[0] / image.size[1]) / 2
            xyxy[2] = xyxy[2] - (IMG_SIZE[0] - IMG_SIZE[0] * image.size[0] / image.size[1]) / 2
 
            xywh = xyxy2xywhn(torch.tensor(xyxy).view(1, 4), 
                              w=IMG_SIZE[0] * image.size[0] / image.size[1], 
                              h=IMG_SIZE[1]
                              ).view(-1).tolist()
        else:
            xywh = xyxy2xywhn(torch.tensor(xyxy).view(1, 4), 
                              w=IMG_SIZE[0],
                              h=IMG_SIZE[1]
                              ).view(-1).tolist()
        # smallerThanAreaThresh: 
        # True -> bounding box larger than area_thresh
        smallerThanAreaThresh = (xywh[2] * xywh[3] * 4032 * 3024 > area_thresh) 
        # onEdge: 
        # True -> bounding box not on the edges (distance to edge larger than edge_thresh )
        notOnEdge =  ((xywh[0] > edge_thresh) 
                   and (xywh[0] < 1 - edge_thresh) 
                   and (xywh[1] > edge_thresh) 
                   and (xywh[1] < 1 - edge_thresh))
        # withinROI: 
        # True -> bounding box is in ROI regioun based on pattern match results
        withinROI = ((xywh[0] * 3024 * math.sin(-theta_s[idx]) * flag[idx] < (xywh[1] * 4032 / kt2 - bt2) * math.sin(-theta_s[idx]) * flag[idx]) 
                     and (xywh[1] * 4032 < kt1 * xywh[0] * 3024 + bt1))
        if smallerThanAreaThresh and notOnEdge and withinROI:
            if save_text:
                label_list.append(('%g ' * 6 + '\n') % (cls, *xywh, conf))
            if save_img:
                color = (255, 0, 0) if conf > CONF_COLOR_THRESH else (0, 255, 255)
                draw = PIL.ImageDraw.Draw(image)
                # add class & conf
                if save_conf:
                    # c = int(cls)
                    # label_txt = f'{CAT_NAMES[c]} {conf:.2f}'
                    label_txt = f'{conf:.2f}'
                else:
                    label_txt = ''
                if cls == 1:
                    shape = [(0,0), (3024,4032)]
                    draw.rectangle(shape, outline=(255,255,0), width=50)
                else:
                    label = [cls, *xywh, conf]
                    draw = draw_bb(draw, [label], color=color, text=label_txt)

    os.makedirs(out_folder, exist_ok=True)
    if save_text & (len(label_list) != 0):
        with open(os.path.join(out_folder, '{}.txt'.format(img_path[:-4])), 'a') as f:
            for line in label_list:
                f.write(line)
    if save_img & (len(label_list) != 0):
        # draw = PIL.ImageDraw.Draw(image)
        draw.line([(0, int(bt1)), (img0.shape[1], int(kt1 * img0.shape[1] + bt1))], fill=(255, 0, 0), width=6)
        draw.line([(-int(bt2), 0), (int(img0.shape[0] / kt2 - bt2), img0.shape[0])], fill=(255, 0, 0), width=6)

        image.save(os.path.join(out_folder, '{}.jpg'.format(img_path[:-4])))

 
def debug(image_folder=IMAGE_FOLDER, template_path=TEMPLATE_PATH, model_path_list=COREML_MODEL):
    
    model_list, model_source = load_model(model_path_list)
        
    template = load_template(template_path)

    time_tracker = {}
    time_sum = 0
    for img_path in tqdm(os.listdir(image_folder)):
        t0 = time.time()
        if (img_path.endswith(".jpg")) and (not img_path.startswith('.')):
            predict(img_path, model_list, image_folder, template, model_source)
        delta_t = time.time() - t0
        time_tracker[img_path] = delta_t
    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item   
    print('Averange process time is {}'.format(time_sum / len(time_tracker)))


def evaluate(image_folder=IMAGE_FOLDER, template_path=TEMPLATE_PATH, model_path_list=COREML_MODEL):
    
    fit_folder = os.path.join(image_folder, 'FIT')
    ok_folder = os.path.join(image_folder, 'OK_Val')

    # Load the model
    model_list, model_source = load_model(model_path_list)

        
    template = load_template(template_path)
    
    def eval (folder):
        print('Trying to evaluate performance on {}'.format(folder))
        img_folder=os.path.join(folder, 'images')
        label_folder=os.path.join(folder, 'labels')
        pred_folder=os.path.join(folder, 'output')
        out_folder=os.path.join(folder, 'eval_output', 'images')
        os.makedirs(pred_folder, exist_ok=True)
        missing = imgPredLabelMatch(img_folder, label_folder, pred_folder)
        if missing:
            print('Predicted labels missing, predicting with the model ... ...')
            for img_path in tqdm(missing):
                predict(img_path, model_list, img_folder, template, model_source, save_img=False)
        return visual_analysis(img_folder, label_folder, pred_folder, out_folder)
    
    df_fit = eval(fit_folder)
    # df_ok = eval(ok_folder)
    
    post_analysis(df_fit, 'FIT')
    cropOverkillEscape(df_fit, os.path.join(fit_folder, 'eval_output', 'images'))
    
    # post_analysis(df_ok, 'OK')
    # cropOverkillEscape(df_ok, os.path.join(ok_folder, 'eval_output', 'images'))



def main(image_folder=IMAGE_FOLDER, template_path=TEMPLATE_PATH, model_path_list=COREML_MODEL):

    # Load the model
    model_list, model_source = load_model(model_path_list)

    
    tfov, tsep = load_template(template_path)

    m = False
    while True:
        if not os.path.isfile(image_folder + '1.jpg'):
            m = True

        if os.path.isfile(image_folder + '1.jpg') & m:
            predict(image_folder + '1.jpg', model_list, image_folder, tfov, tsep)
            m = False

        else:
            time.sleep(0.5)

if MODE == 'debug':
    debug()
elif MODE == 'eval':
    evaluate()
else:
    main()

