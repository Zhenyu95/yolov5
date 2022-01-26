from coremltools.models import model
from utils.general import scale_coords, non_max_suppression, xyxy2xywh, xywhn2xyxy, xyxy2xywhn
import coremltools
import torch
import numpy as np
import random
import os
import PIL.Image
import PIL.ImageDraw 
import shutil
import time
from tqdm import tqdm

# Parameters and Global variables
MODE = 'debug'
SAVE_IMG = False
VIEW_IMG = False
SAVE_TXT = True
CAT_NAMES = ['Screw']

# Anchor box can be checked in pytorch model
# ANCHORS = ([2.375,3.375, 5.5,5, 4.75,11.75], 
#            [6,4.25, 5.375,9.5, 11.25,8.5625], 
#            [4.375,9.40625, 9.46875,8.25, 7.43750,16.93750],
#            [6.81250,9.60938, 11.54688,5.93750, 14.45312,12.37500])
ANCHORS = ([1.25000,  1.62500, 2.00000,  3.75000, 4.12500,  2.87500], 
           [1.87500,  3.81250, 3.87500,  2.81250, 3.68750,  7.43750], 
           [3.62500,  2.81250, 4.87500,  6.18750, 11.65625, 10.18750])
# stide can be check in pytorch model
# stride = [8, 16, 32, 64]
stride = [8, 16, 32]
# target size of input image (width, height)
# IMG_SIZE = (960, 1280)
# IMG_SIZE = (1440, 1920)
# IMG_SIZE = (1472, 1920)
# IMG_SIZE = (2304, 3072)
IMG_SIZE = (1920, 2560)
# confidence threshold
conf_thres = .35
area_thres = 300
edge_thres = 0.002

# Params and global variables that should be kept as it is
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CAT_NAMES))]
PATH = "./"
nc = len(CAT_NAMES)
nl = len(ANCHORS)
na = len(ANCHORS[0]) // 2
no = nc + 5
# initiate grid and anchor_grid
grid = [torch.zeros(1)] * nl
a = torch.tensor(ANCHORS).float().view(nl, -1, 2)
anchor_grid = [torch.zeros(1)] * nl



if MODE == 'debug':
    COREML_MODEL = ['/Users/zhenyu/Downloads/station1_0114.mlmodel',
                    # '/Users/zhenyu/Desktop/runs/train/exp2/weights/best.mlmodel',
                    # '/Users/zhenyu/Desktop/runs/train/exp3/weights/best.mlmodel'
                    ]
    # IMAGE_FOLDER = "/Users/zhenyu/Library/CloudStorage/Box-Box/MLProject:IphoneAOI/datasets/After T-Cowling/NG/"
    IMAGE_FOLDER = '/Users/zhenyu/Desktop/dataset_test/images/test/'
    # IMAGE_FOLDER = '/Users/zhenyu/Library/CloudStorage/Box-Box/MLProject:IphoneAOI/datasets/After T-Cowling/NG_Image/FOV4_Roy/'
    OUT_FOLDER = "/Users/zhenyu/Desktop/pred/"
    # OUT_FOLDER = '/Users/zhenyu/Library/CloudStorage/Box-Box/MLProject:IphoneAOI/datasets/After T-Cowling/Overkill/Results_289/FOV4_31_conf'
else:
    COREML_MODEL = ['/Users/iphoneaoi/Documents/yolov5/best1.mlmodel',
                    '/Users/iphoneaoi/Documents/yolov5/best2.mlmodel',
                    '/Users/iphoneaoi/Documents/yolov5/best3.mlmodel',
                    ]
    IMAGE_FOLDER = "/Users/iphoneaoi/Documents/yolov5/images/"
    OUT_FOLDER = "/Users/iphoneaoi/Documents/yolov5/runs/detect/"

def make_grid(nx=20, ny=20, i=0):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    anchor_grid = (a[i] * stride[i]).view((1, na, 1, 1, 2)).expand(1, na, ny, nx, 2).float()
    return grid, anchor_grid

def resize_image(source_image):
    background = PIL.Image.new('RGB', IMG_SIZE, "black")
    source_image.thumbnail(IMG_SIZE)
    (w, h) = source_image.size
    background.paste(source_image, (int((IMG_SIZE[0] - w) / 2), int((IMG_SIZE[1] - h) / 2 )))
    return background

def eval(image, model, file_name):
    resized = resize_image(image.copy())

    predictions = model.predict({'image': resized})

    z = []  # inference output
    x = []
    # for head in ['var_1763', 'var_1778', 'var_1793', 'var_1808']:
    # for head in ['var_1625', 'var_1640', 'var_1655']:
    # for head in ['var_1295', 'var_1308', 'var_1321']:
    # for head in ['var_2093', 'var_2108', 'var_2123', 'var_2138']:
    for head in ['var_1383', 'var_1396', 'var_1409']:
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


def pred(img_path, model_list):
    image = PIL.Image.open(os.path.join(IMAGE_FOLDER, img_path))
    pred = torch.tensor([])
    for model in model_list:
        pred = torch.cat((pred, eval(image, model, img_path)), 1)
    nms = non_max_suppression(pred, conf_thres, .3, classes=None, agnostic=False)[0]
    label=[]
    for *xyxy, conf, cls in nms:
        xywh = xyxy2xywhn(torch.tensor(xyxy).view(1, 4), w=IMG_SIZE[0], h=IMG_SIZE[1]).view(-1).tolist()
        if (xywh[2]*xywh[3]*4032*3024 > area_thres) and (xywh[0] > edge_thres) and (xywh[0] < 1-edge_thres) and (xywh[1] > edge_thres) and (xywh[1] < 1-edge_thres):
            if SAVE_TXT:
                label.append(('%g ' * 6 + '\n') % (cls, *xywh, conf))
            if SAVE_IMG:
                draw = PIL.ImageDraw.Draw(image)
                draw.rectangle(np.array(torch.tensor(xyxy).view(2,2)*2.1), outline='red', width=6)
                font = PIL.ImageFont.truetype("SFCompact.ttf", 70)
                draw.text(np.array(torch.tensor(xyxy).view(2,2)*2.1)[0], str(conf)[7:12], fill ="red", font=font)
    if SAVE_TXT and (len(label)!=0):
        with open(os.path.join(OUT_FOLDER, '{}.txt'.format(img_path[:-4])), 'a') as f:
            for line in label:
                f.write(line)
        # with open(os.path.join(OUT_FOLDER, img_path.split('/')[0], 'classes.txt'), 'a') as f:
        #     f.write('screw\n')
    if SAVE_IMG:
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
    for img_path in tqdm(os.listdir(IMAGE_FOLDER)):
        t0 = time.time()
        if img_path.endswith(".jpg"):
            pred(img_path, model_list)
        delta_t = time.time() - t0
        time_tracker[img_path] = delta_t
    for key, item in time_tracker.items():
        # print('{} takes {} seconds'.format(key, item))
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
        # print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum/len(time_tracker)))    
    
    
def main():
    model_list = []
    
    # Load the model
    for each in COREML_MODEL:
        print('Loading model {}...'.format(each))
        model_list.append(coremltools.models.model.MLModel(each))
        print('Model {} loaded'.format(each))

    m=False
    while True:
        if not os.path.isfile(IMAGE_FOLDER+'1.jpg'):
            m = True
        
        if os.path.isfile(IMAGE_FOLDER+'1.jpg') & m:
            pred(IMAGE_FOLDER+'1.jpg', model_list
                 )
            m=False
        
        else:
            time.sleep(0.5)
            
if MODE == 'debug':
    debug()
else:
    main()