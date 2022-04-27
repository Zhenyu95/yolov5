from ast import Raise
from distutils.log import error
import torch
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import math
import torch.nn as nn
from torchvision import transforms
import cv2
import pandas as pd
import os
import re
from yaml import Loader, load, dump
from tqdm import tqdm
import coremltools


# load all models from model_path_list
# and check the source of the model
# 'tf'    -> model converted from BJML teams tensorflow model
# 'torch' -> model converted from official pytorch YOLOv5 model
def load_model(model_path_list):
    model_list = []
    model_source = []
    # Load the model
    for each in model_path_list:
        print('Loading model {}...'.format(each))
        model = coremltools.models.model.MLModel(each)
        model_list.append(model)
        print('Model {} loaded'.format(each))
        model_source.append(str(model.input_description))
    
    model_source = set(model_source)
    if len(model_source) != 1:
        raise Exception('Model type does not match')
    else:
        if model_source == set(['Features(image)']):
            return model_list, 'torch'
        else:
            return model_list, 'tf'
        

# helper function to decode the model output
def make_grid(nx, ny, i, a, stride, na):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    anchor_grid = (a[i] * stride[i]).view((1, na, 1, 1, 2)).expand(1, na, ny, nx, 2).float()
    return grid, anchor_grid


# resize the image to target size
def resize_image(source_image, img_size):
    background = PIL.Image.new('RGB', img_size, "black")
    source_image.thumbnail(img_size)
    (w, h) = source_image.size
    background.paste(source_image, (int((img_size[0] - w) / 2), int((img_size[1] - h) / 2)))
    return background


def mean(x):
    y = np.sum(x) / np.size(x)
    return y


# normalized cross correlation
def corr(a, b):
    a = a - mean(a)
    b = b - mean(b)

    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


# load template images from template_path
def load_template(template_path, firstTime=False):
    # Load Template Image
    tfov = []
    for i in range(4):
        # Temp = cv2.imread(template_path + '/T' + str(i + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
        Temp = cv2.imread(os.path.join(template_path, 'T{}.jpg'.format(i + 1)), cv2.IMREAD_GRAYSCALE)
        tfov.append(Temp)

    # Load FOV related template
    tsep = []
    for i in range(4):
        # T1 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T1.jpg', cv2.IMREAD_GRAYSCALE)
        T1 = cv2.imread(os.path.join(template_path, 'FOV{}'.format(i + 1), 'T1.jpg'), cv2.IMREAD_GRAYSCALE)
        # T2 = cv2.imread(template_path + '/FOV' + str(i + 1) + '/T2.jpg', cv2.IMREAD_GRAYSCALE)
        T2 = cv2.imread(os.path.join(template_path, 'FOV{}'.format(i + 1), 'T2.jpg'), cv2.IMREAD_GRAYSCALE)
        tsep.append([T1, T2])

    if firstTime:
        idxs, xy = generateYaml(template_path)
    else:
        try:
            with open(os.path.join(template_path, 'template_idxs.yaml')) as f:
                idxs, xy = load(f, Loader=Loader)
        except FileNotFoundError:
            idxs, xy = generateYaml(template_path)

    return tfov, tsep, idxs, xy


# pattern match, get idxs and xy
def generateYaml(file_path, fov_list=['FOV1', 'FOV2', 'FOV3', 'FOV4']):
    idxs, xy = generate_template(file_path, fov_list)

    with open(os.path.join(file_path, 'template_idxs.yaml'), 'w') as f:
        dump((idxs, xy), f)

    return idxs, xy


# pattern match, get FOVid
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


# pattern match, get the coordinates
def GetMatchTemplate(img0, idxtt1, idxtt2, template, n_point=5):
    f = nn.AvgPool2d(4, stride=4)
    trans = transforms.Compose([
        transforms.ToTensor(), ])

    # crop the image to get the search region
    imgt1 = img0[idxtt1[0]:idxtt1[1], idxtt2[0]:idxtt2[1]]
    imgt01 = trans(imgt1)
    imgt01 = imgt01[None]
    # scaling image to speed up
    imgt01 = f(imgt01)
    imgt01 = torch.squeeze(imgt01).numpy() * 255
    imgt01 = imgt01.astype(np.uint8)

    t1 = template
    t01 = torch.squeeze(f(trans(t1))) * 255
    t01 = np.array(t01, dtype=np.uint8)
    # do template match roughly
    res = cv2.matchTemplate(imgt01, t01, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0] * 4, max_loc[1] * 4

    xs1 = k1 - n_point if k1 > n_point else 0
    ys1 = h1 - n_point if h1 > n_point else 0

    xe1 = k1 + t1.shape[1] + n_point if k1 + t1.shape[1] + n_point < imgt1.shape[1] else imgt1.shape[1]
    ye1 = h1 + t1.shape[0] + n_point if h1 + t1.shape[0] + n_point < imgt1.shape[0] else imgt1.shape[0]

    imgt1 = imgt1[ys1:ye1, xs1:xe1]
    # do pixel level template match
    res = cv2.matchTemplate(imgt1, t1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    k1, h1 = max_loc[0], max_loc[1]

    return xs1 + k1, ys1 + h1


# pattern match, get 2 orthogonal line, which circles the ROI
def GetROILine(image, idx, tsep, idxs, xy, theta_s):
    # PIL image read cost time
    # get actual ROI x line & y line
    t1, t2 = tsep[idx]
    idxt1, idxt2, idxt3, idxt4 = idxs[:, idx]
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

    return kt1, kt2, bt1, bt2, img0


# convert pytorch tensor to PIL image
def Tensor2Image(img1):

    img1 = torch.squeeze(img1).numpy()
    img1 = img1.transpose((1, 2, 0))
    img1 = np.array(img1 * 255, dtype=np.uint8)
    img2 = PIL.Image.fromarray(img1)

    return img2


# code copied from our beloved Steve's GenerateTemplate.py
def generate_template(file_path, fov_list):

    f1 = nn.AvgPool2d(2, stride=2)
    f2 = nn.AvgPool2d(16, stride=16)
    trans = transforms.Compose([transforms.ToTensor(), ])

    idx1, idx2, idx3, idx4 = [], [], [], []
    xy = []

    for i, fn in enumerate(fov_list):
        img0 = PIL.Image.open(file_path + '/' + fn + '.jpg')
        img0 = trans(img0)
        img0 = img0[None]

        imgt = f2(img0)
        imgt = Tensor2Image(imgt)
        imgt.save(file_path + '/T' + str(i + 1) + '.jpg', quality=100)

        _, _, m, n = img0.shape
        Gain = [1, n, m, n, m]

        # read txt
        df_bw = pd.read_csv(file_path + '/' + fn + '.txt', sep=' ', header=None,
                            names=['class', 'x', 'y', 'w', 'h'], index_col=False)
        df_1 = df_bw[df_bw['class'].isin([0])].reset_index(drop=True)
        df_2 = df_bw[df_bw['class'].isin([1])].reset_index(drop=True)

        df_1 = df_1 * Gain
        df_2 = df_2 * Gain
        # print(df_1)
        os.makedirs(file_path + '/' + fn, exist_ok=True)

        for j in range(len(df_1)):
            imgt0 = img0[:, :, int(df_1.loc[j, 'y'] - df_1.loc[j, 'h'] / 2):int(df_1.loc[j, 'y'] + df_1.loc[j, 'h'] / 2) + 1,
                         int(df_1.loc[j, 'x'] - df_1.loc[j, 'w'] / 2):int(df_1.loc[j, 'x'] + df_1.loc[j, 'w'] / 2) + 1]

            imgt1 = Tensor2Image(imgt0)
            imgt1.save(file_path + '/' + fn + '/T' + str(j + 1) + '.jpg', quality=100)

            imgt2 = f1(imgt0)
            imgt3 = Tensor2Image(imgt2)
            imgt3.save(file_path + '/' + fn + '/T0' + str(j + 1) + '.jpg', quality=100)

            if j == 0:
                idx1.append((int(df_2.loc[j, 'y'] - df_2.loc[j, 'h'] / 2),
                            int(df_2.loc[j, 'y'] + df_2.loc[j, 'h'] / 2) + 1))
                idx2.append((int(df_2.loc[j, 'x'] - df_2.loc[j, 'w'] / 2),
                            int(df_2.loc[j, 'x'] + df_2.loc[j, 'w'] / 2) + 1))
            else:
                idx3.append((int(df_2.loc[j, 'y'] - df_2.loc[j, 'h'] / 2),
                            int(df_2.loc[j, 'y'] + df_2.loc[j, 'h'] / 2) + 1))
                idx4.append((int(df_2.loc[j, 'x'] - df_2.loc[j, 'w'] / 2),
                            int(df_2.loc[j, 'x'] + df_2.loc[j, 'w'] / 2) + 1))
            xy.append([imgt1.width, imgt1.height])
    mask = np.array([[1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0]])
    offset = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
    xy = (np.array(xy).reshape((4, 4))) * mask + offset
    idxs = np.array([idx1, idx2, idx3, idx4])
    return idxs, xy


# to get idx1, idx2, idx3, idx4 and pack into a tuple idxs
def get_idxs(new=False, file_path=None):
    if new:
        idxs, xy = generateYaml(file_path)
    else:
        try:
            with open(os.path.join(file_path, 'template_idxs.yaml')) as f:
                idxs, xy = load(f, Loader=Loader)
        except FileNotFoundError:
            idxs, xy = generateYaml(file_path)
    return idxs, xy


# calculate iou
def iou_calc(box1, box2, eps=1e-7):

    box2 = box2.T

    b1_x1, b1_x2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b1_y1, b1_y2 = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2
    b2_x1, b2_x2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    b2_y1, b2_y2 = box2[2] - box2[4] / 2, box2[2] + box2[4] / 2

    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # Union Area
    w1, h1 = box1[3], box1[4] + eps
    w2, h2 = box2[3], box2[4] + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou


# draw bounding box and corresponding text on img
def draw_bb(img, label, color=(0, 0, 255), text='', lineWidth=6, padding=16):

    for box in label:
        try:
            _, x, y, w, h = box
        except ValueError:
            _, x, y, w, h, conf = box

        if text == 'conf':
            label_text = f'{conf:.2f}'
        else:
            label_text = text

        width, height = img.im.size

        x_min = int((x - w / 2) * width)
        y_min = int((y - h / 2) * height)
        x_max = int((x + w / 2) * width)
        y_max = int((y + h / 2) * height)

        bb_position = np.array((x_min, y_min, x_max, y_max))

        bb_offset = np.array((-padding, -padding, padding, padding))

        img.rectangle((bb_position + bb_offset).tolist(), outline=color, width=lineWidth)

        font = PIL.ImageFont.truetype('Arial.ttf', max(round(width * 0.01), 32))
        w_text, h_text = font.getsize(label_text)

        text_position = np.array((x_min - padding, y_min - h_text - padding))
        if text_position[1] <= 0:
            text_position[1] = y_max + padding
        text_position_br = text_position + [w_text, h_text]
        text_position = text_position - np.minimum((text_position), (0, 0)) - \
            np.maximum((text_position_br - [width, height]), (0, 0))

        text_bg = np.array((text_position, text_position + [w_text, h_text]))
        text_bg.resize((4,))
        img.rectangle(text_bg.tolist(), fill=color)
        img.text(text_position.tolist(), label_text, fill='white', font=font)
    return img


# read image's label from target folder and return sorted array
def get_label(image, label_folder):
    try:
        with open(os.path.join(label_folder, '{}.txt'.format(image[:-4]))) as f:
            label = [list(map(float, line.rstrip().split())) for line in f]
    except FileNotFoundError:
        label = []
    label.sort()
    label = np.array(label)
    return label


def visual_analysis(img_folder, label_folder, pred_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    df_result = pd.DataFrame(columns=['image', 'FOV', 'EscapeOrNot', 'OverkillOrNot',
                             'Label', 'Prediction', 'TP', 'Overkills', 'Escapes'])

    for image in tqdm([img for img in os.listdir(img_folder) if img.endswith('.jpg')]):

        label = get_label(image, label_folder)
        pred = get_label(image, pred_folder)

        n_label = label.shape[0]
        index_label_list, index_pred_list, label_TP = [], [], []

        img = PIL.Image.open(os.path.join(img_folder, image))
        draw = PIL.ImageDraw.Draw(img)
        for index_label, value in enumerate(label):
            iou = (iou_calc(value, pred) > 0.01).nonzero()[0]
            # for each in index_pred:
            #     label_TP.append(pred[each])
            #     break
            if np.size(iou):
                index_label_list.append(index_label)
        for index_pred, value in enumerate(pred):
            iou = (iou_calc(value, label) > 0.01).nonzero()[0]
            if np.size(iou):
                index_pred_list.append(index_pred)
        label_TP = pred[list(set(index_pred_list))]
        pred = np.delete(pred, list(set(index_pred_list)), 0)
        label = np.delete(label, list(set(index_label_list)), 0)

        draw = draw_bb(draw, label_TP, color=(0, 0, 128), text='')
        draw = draw_bb(draw, pred, color=(0, 215, 255), text='OVERKILL')
        draw = draw_bb(draw, label, color=(139, 0, 139), text='ESCAPE')

        if np.any(label) or np.any(pred):
            img.save(os.path.join(out_folder, image))

        result = {'image': image,
                  'Label': n_label,
                  'Prediction': len(label_TP) + len(pred),
                  'TP': len(label_TP),
                  'Overkills': np.array2string(pred, separator=','),
                  'Escapes': np.array2string(label, separator=',')}
        df_result = df_result.append(result, ignore_index=True)

    df_result.FOV = df_result.apply(lambda row: getFOVFromName(row.image), axis=1)
    df_result.EscapeOrNot = df_result.apply(lambda row: row.TP != row.Label, axis=1)
    df_result.OverkillOrNot = df_result.apply(lambda row: row.TP != row.Prediction, axis=1)
    df_result.to_csv(os.path.join(out_folder, '../result.csv'))

    return df_result


def getFOVFromName(image):
    for fov in ['FOV1', 'FOV2', 'FOV3', 'FOV4']:
        if fov in image:
            return fov


def imgPredLabelMatch(img_folder, label_folder, pred_folder):
    try:
        label_list = [label[:-4] for label in os.listdir(label_folder) if label.endswith('.txt')]
    except FileNotFoundError:
        label_list = []
    try:
        pred_list = [pred[:-4] for pred in os.listdir(pred_folder) if pred.endswith('.txt')]
    except FileNotFoundError:
        pred_list = []
        
    check = [f'{item}.jpg' for item in label_list if (item not in pred_list) and (item != 'classes')]
    
    if (not label_list) and (not pred_list):
        check = [img for img in os.listdir(img_folder) if img.endswith('.jpg')]

    return check


def string2nparray(string):
    decode = re.split(r'\]|\[', string)
    decode = list(filter(lambda x: len(x)>10, decode))
    array = np.array([list(map(float, line.rstrip().split(','))) for line in decode])
    return array


def array2cord(array):
    try:
        cord = array[:, 1:3]
        return cord
    except IndexError:
        return []


def post_analysis(df, text=''):
    n_TP = df.TP.sum()
    n_pred = df.Prediction.sum()
    n_label = df.Label.sum()
    screw_precision = n_TP / n_pred
    try:
        screw_recall = n_TP / n_label
    except:
        screw_recall = 0
    
    image_escape_rate = df.EscapeOrNot.sum()/df.EscapeOrNot.count()
    image_overkill_rate = df.OverkillOrNot.sum()/df.OverkillOrNot.count()
    
    print('*'*88)
    print('Here is the evaluation result of {} images: \n'.format(text))
    print('-'*38+'Screw Level'+'-'*39)
    print('Screw Level Precision     : {:%}'.format(screw_precision))
    print('Screw Level Capture Rate  : {:%}'.format(screw_recall))
    print('Screw Level Escape Rate   : {:%}'.format(1-screw_recall))
    print('-'*38+'Screw Level'+'-'*39)
    print('Image Level Capture Rate  : {:%}'.format(1-image_escape_rate))
    print('Image Level Escape Rate   : {:%}'.format(image_escape_rate))
    print('Image Level Overkill Rate : {:%}'.format(image_overkill_rate))
    print('*'*88)
    
    
def cropOverkillEscape(df, image_folder):
    
    df.Overkills = df.Overkills.apply(lambda col: string2nparray(col))
    df.Escapes = df.Escapes.apply(lambda col: string2nparray(col))
    df['escapesCord'] = df.Escapes.apply(lambda col: array2cord(col))
    df['overkillCord'] = df.Overkills.apply(lambda col: array2cord(col))
    
    
    def crop(image_name, xy, image_folder, out_folder):
        image = PIL.Image.open(os.path.join(image_folder, image_name))
        xy = xy * [image.width, image.height]
        xyxy = np.append(xy, xy, axis=1)
        xyxy = xyxy + [-256, -256, 256, 256]
        for index, cord in enumerate(xyxy):
            cropped = image.crop(cord)
            cropped.save(os.path.join(out_folder,'{}_{}.jpg'.format(image_name[:-4], index)))
            
            
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if np.array(row.escapesCord).any():
            out_folder = os.path.join(image_folder, '../escape_crop')
            os.makedirs(out_folder, exist_ok=True)
            crop(row.image, row.escapesCord, image_folder, out_folder)
        if np.array(row.overkillCord).any():
            out_folder = os.path.join(image_folder, '../overkill_crop')
            os.makedirs(out_folder, exist_ok=True)
            crop(row.image, row.overkillCord, image_folder, out_folder)
            

