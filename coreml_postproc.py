from utils.general import scale_coords, non_max_suppression, xyxy2xywh, save_one_box
import coremltools
import torch
import numpy as np
import random
import os
import PIL.Image
import shutil
import time

MODE = 'debug'
SAVE_IMG = False
VIEW_IMG = False
SAVE_TXT = True
CAT_NAMES = ['Screw', 'unknown']
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CAT_NAMES))]
PATH = "./"
ANCHORS = ([116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]) # from <model>.yml
IMG_SIZE = (2560, 2560)
nc = len(CAT_NAMES)
nl = len(ANCHORS)
na = len(ANCHORS[0]) // 2
no = nc + 5  # number of outputs per anchor
grid = [torch.zeros(1)] * nl  # init grid
a = torch.tensor(ANCHORS).float().view(nl, -1, 2)
anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2)
stride = [32, 16, 8] # check your model config
conf_thres = .3


if MODE == 'debug':
    COREML_MODEL = "/Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/best_1106.mlmodel"
    IMAGE_FOLDER = "/Users/zhenyu/Desktop/val/"
    OUT_FOLDER = "/Users/zhenyu/Desktop/test/"
else:
    COREML_MODEL = "/Users/iphoneaoi/Documents/yolov5/best_1106.mlmodel"
    IMAGE_FOLDER = "/Users/iphoneaoi/Documents/yolov5/images/"
    OUT_FOLDER = "/Users/iphoneaoi/Documents/yolov5/runs/detect/"


def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def resize_image(source_image):
    # background = source_image.crop((0,0, 2560, 2560))
#TODO: BUG HERE!!!!!!!!!!!!!!
    background = PIL.Image.new('RGB', IMG_SIZE, "black")
    source_image.thumbnail(IMG_SIZE)
    (w, h) = source_image.size
    background.paste(source_image, (int((IMG_SIZE[0] - w) / 2), int((IMG_SIZE[1] - h) / 2 )))
    return background

def eval(file_name):   
    # image = resize_image(image)
    # _, img = load_image(IMAGE_FOLDER+file_name, resize_to=None)
    
    source = PIL.Image.open(os.path.join(IMAGE_FOLDER, file_name))
    cropped = source.crop((0,0,2560,2560))
    # cropped.save('{}{}_cropped.jpg'.format(IMAGE_FOLDER,file_name[:-4]))
    # img = PIL.Image.open('{}{}_cropped.jpg'.format(IMAGE_FOLDER,file_name[:-4]))
    # resized = resize_image(source)
    # image0shape = np.array(image).astype(np.float32).shape
    
    # img_np = np.array(img).astype(np.float32)
#TODO: BUG HERE!!!!!!!!!!!!!!!
    # img = torch.zeros((1,3,IMG_SIZE[0],IMG_SIZE[1]))
    # img[0, :, :, :] = torch.Tensor(np.array(resized)).permute(2, 0, 1)
    # im0 = np.array(source)

    predictions = model.predict({'image': cropped})

    z = []  # inference output
    x = []
    for pred in predictions:
        x.append(torch.Tensor(predictions[pred]))
    x.reverse()

    for i in range(nl):
        bs, _, ny, nx, _ = x[i].shape

        if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = make_grid(nx, ny)

        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))

    pred = (torch.cat(z, 1), x)[0]

    pred = non_max_suppression(pred, conf_thres, .5, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s = "./", ""

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, CAT_NAMES[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                if SAVE_TXT:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / 1280).view(-1).tolist()  # normalized xywh
                    with open(os.path.join(OUT_FOLDER, 'result_1.txt'), 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
        cropped.save(os.path.join(OUT_FOLDER, 'result_1.jpg'))
# TODO: BUG HERE!!!!!!!!!!!
    #             if SAVE_IMG or VIEW_IMG:  # Add bbox to image
    #                 label = '%s %.2f' % (CAT_NAMES[int(cls)], conf)
    #                 save_one_box(xyxy, im0, file='label.jpg', BGR=True)

    # if SAVE_IMG:
    #     cv2.imwrite(os.path.join(OUT_FOLDER, file_name), im0)

def debug():
    global model
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

    # Load the model
    model = coremltools.models.model.MLModel(COREML_MODEL)

    eval('2.jpg')
    
    
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