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
SAVE_IMG = True
VIEW_IMG = False
SAVE_TXT = True
ENSEMBLE = False
CAT_NAMES = ['Screw', 'unknown']

# Anchor box can be checked in pytorch model
ANCHORS = ([2.375,3.375, 5.5,5, 4.75,11.75], 
           [6,4.25, 5.375,9.5, 11.25,8.5625], 
           [4.375,9.40625, 9.46875,8.25, 7.43750,16.93750],
           [6.81250,9.60938, 11.54688,5.93750, 14.45312,12.37500])
# stide can be check in pytorch model
stride = [8, 16, 32, 64]
# target size of input image (width, height)
IMG_SIZE = (2304, 3072)
# confidence threshold
conf_thres = .2


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
    COREML_MODEL = ["/Users/zhenyu/Box/MLProject:IphoneAOI/weights/yolov5l6_3072*2304_20211116/weights/best.mlmodel",
                    "/Users/zhenyu/Box/MLProject:IphoneAOI/weights/yolov5l6_3072*2304_20211124/weights/best.mlmodel"]
    IMAGE_FOLDER = "/Users/zhenyu/Desktop/val/"
    OUT_FOLDER = "/Users/zhenyu/Desktop/test/"
else:
    COREML_MODEL = "/Users/iphoneaoi/Documents/yolov5/best_1106.mlmodel"
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
    for head in ['var_1763', 'var_1778', 'var_1793', 'var_1808']:
        x.append(torch.Tensor(predictions[head]))

    for i in range(nl):
        bs, _, ny, nx, _ = x[i].shape

        # if grid[i].shape[2:4] != x[i].shape[2:4]:
        grid[i], anchor_grid = make_grid(nx, ny, i)

        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
        z.append(y.view(bs, -1, no))
    
    pred = torch.cat(z, 1)
    
    return pred


def debug():
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

    model_list = []
    
    # Load the model
    for each in COREML_MODEL:
        model_list.append(coremltools.models.model.MLModel(each))
    time_tracker = {}
    time_sum = 0
    for img_path in tqdm(os.listdir(IMAGE_FOLDER)):
        if img_path.endswith(".jpg") and not img_path.startswith('.'):
            image = PIL.Image.open(os.path.join(IMAGE_FOLDER, img_path))
            draw = PIL.ImageDraw.Draw(image)
            pred = torch.tensor([])
            t0 = time.time()
            for model in model_list:
                pred = torch.cat((pred, eval(image, model, img_path)), 1)
            nms = non_max_suppression(pred, conf_thres, .3, classes=None, agnostic=False)[0]
            label=[]
            for *xyxy, _, cls in nms:
                if SAVE_TXT:
                    xywh = xyxy2xywhn(torch.tensor(xyxy).view(1, 4), w=IMG_SIZE[0], h=IMG_SIZE[1]).view(-1).tolist()
                    label.append(('%g ' * 5 + '\n') % (cls, *xywh))
                if SAVE_IMG:
                    draw.rectangle(np.array(torch.tensor(xyxy).view(2,2)*1.3125), outline='red', width=6)
            if SAVE_TXT:
                with open(os.path.join(OUT_FOLDER, '{}.txt'.format(img_path[:-4])), 'a') as f:
                    for line in label:
                        f.write(line)
            if SAVE_IMG:
                image.save(os.path.join(OUT_FOLDER, '{}.jpg'.format(img_path[:-4])))
            delta_t = time.time() - t0
            time_tracker[img_path] = delta_t
        break
    for key, item in time_tracker.items():
        print('{} takes {} seconds'.format(key, item))
        time_sum += item
    print('Averange process time is {}'.format(time_sum/len(time_tracker)))
    
    
def main():
    model = coremltools.models.model.MLModel(COREML_MODEL)
    print('model 1 loaded')

    m=False
    while True:
        if not os.path.isfile(IMAGE_FOLDER+'1.jpg'):
            m = True
        
        if os.path.isfile(IMAGE_FOLDER+'1.jpg') & m:
            eval(IMAGE_FOLDER+'1.jpg')
            m=False
        
        else:
            time.sleep(0.5)
            
if MODE == 'debug':
    debug()
else:
    main()