import os
import numpy as np
from tqdm import tqdm
import cv2
import torch

IMG_FOLDER = '/Users/zhenyu/Desktop/AOI7/Real/images/'
LABEL_FOLDER = '/Users/zhenyu/Desktop/AOI7/Real/labels/'
PRED_FOLDER = '/Users/zhenyu/Desktop/AOI7/Real/images/Output/'
OUT_FOLDER = '/Users/zhenyu/Desktop/overkill/'


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


def draw_bb_iou(img, label, color=(0, 0, 255), text=''):

    img_h, img_w = img.shape[:2]

    for box in label:
        x, y, w, h = box[1:]

        x_min = int((x - w / 2) * img_w)
        y_min = int((y - h / 2) * img_h)
        x_max = int((x + w / 2) * img_w)
        y_max = int((y + h / 2) * img_h)
        img = cv2.rectangle(img, (x_min - 6, y_min - 6), (x_max + 6, y_max + 6), color=color, thickness=4)
        cv2.putText(img, text, (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6)
#         cv2.putText(img, text, (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

    return img


n_label = 0
n_pred = 0
n_FP = 0
n_FN = 0
TP = []
FN = []
FOV = {'FOV1': 0, 'FOV2': 0, 'FOV3': 0, 'FOV4': 0}

for image in tqdm([img for img in os.listdir(IMG_FOLDER) if img.endswith('.jpg')]):
    try:
        with open(os.path.join(LABEL_FOLDER, '{}.txt'.format(image[:-4]))) as f:
            label = [list(map(float, line.rstrip().split())) for line in f]
    except FileNotFoundError:
        label = []
    try:
        with open(os.path.join(PRED_FOLDER, '{}.txt'.format(image[:-4]))) as f:
            pred = [list(map(float, line.rstrip().split())) for line in f]
    except FileNotFoundError:
        pred = []
    label.sort()
    pred.sort()
    label = np.array(label)
    n_label += label.shape[0]
    pred = np.array(pred)
    n_pred += pred.shape[0]
    index_label_list = []

    img = cv2.imread(os.path.join(IMG_FOLDER, image))
    for index_label, value in enumerate(label):
        iou = iou_calc(value, pred)
        try:
            iou_max, index_pred = torch.max(torch.tensor(iou), 0)
        except IndexError:
            continue
        if iou_max >= 1e-5:
            if pred[index_pred][-1] >= 0.5:
                img = draw_bb_iou(img, [value], color=(0, 0, 128))
                TP.append(pred[index_pred][5])
            elif pred[index_pred][-1] >= 0.3:
                img = draw_bb_iou(img, [value], color=(255, 144, 30))
                TP.append(pred[index_pred][5])
            pred = np.delete(pred, index_pred, 0)
            index_label_list.append(index_label)
# uncomment if need to count label numbers in each FOV
#     for fov in ['FOV1', 'FOV2', 'FOV3', 'FOV4']:
#         if fov in image:
#             FOV[fov] += len(label)

    label = np.delete(label, index_label_list, 0)

    n_FN += label.shape[0]
    n_FP += pred.shape[0]

    img = draw_bb_iou(img, pred[:, :-1], color=(0, 215, 255), text='OVERKILL')
    img = draw_bb_iou(img, label, color=(139, 0, 139), text='ESCAPE')
# uncomment if need to count escaped in each FOV
#     if np.any(label):
#         for fov in ['FOV1', 'FOV2', 'FOV3', 'FOV4']:
#             if fov in image:
#                 FOV[fov] += len(label)
#                 break

#     if np.any(label) or np.any(pred):
#         cv2.imwrite(os.path.join(OUT_FOLDER, '{}_iou.jpg'.format(image[:-4])), img)
print('Precision is {:%}'.format((n_pred - n_FP) / n_pred))
print('Recall is {:%}'.format((n_pred - n_FP) / n_label))
print('Escape rate is {:%}'.format(1 - (n_pred - n_FP) / n_label))
print('Overkill rate is {:%}'.format((n_FP) / n_label))
