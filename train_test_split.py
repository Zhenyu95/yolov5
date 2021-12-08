import os
import random
import shutil
import string
import cv2
import numpy as np
from tqdm import tqdm

RAW_IMG_PATH = '/Users/zhenyu/Documents/Scripts/BatteryAOI/Image_Generation_Python_Code/Output/Images/'
RAW_LABEL_PATH = '/Users/zhenyu/Documents/Scripts/BatteryAOI/Image_Generation_Python_Code/Output/Yolo_Format/'
BLANK_IMG_PATH = ''
IMG_DST = '/Users/zhenyu/Library/CloudStorage/Box-Box/MLProject:IphoneAOI/datasets/data_4032*3024_20211205/images/'
LABEL_DST = '/Users/zhenyu/Library/CloudStorage/Box-Box/MLProject:IphoneAOI/datasets/data_4032*3024_20211205/labels/'
TRAIN = 'train/'
TEST = 'test/'
# IMG_DST = '/Users/zhenyu/Desktop/test/'
# LABEL_DST = '/Users/zhenyu/Desktop/test/'
# TRAIN = ''
# TEST = ''
BACKGROUND_PATH = ['/Users/zhenyu/Desktop/data/OK images/left/', 
                   '/Users/zhenyu/Desktop/data/OK images/right/']
RATIO = 0.8
BACKGROUND_NUMBER = 300
CROP_SIZE = 2560


def get_lists(raw_img_path=RAW_IMG_PATH, raw_label_path=RAW_LABEL_PATH):
    img_list = [f for f in os.listdir(raw_img_path) 
                if (os.path.isfile(os.path.join(raw_img_path, f)) and f.endswith('.jpg'))]
    label_list = [f for f in os.listdir(raw_label_path) 
                if (os.path.isfile(os.path.join(raw_label_path, f)) and f.endswith('.txt'))]
    return img_list, label_list


def crop_img(img_path, name, img_dst, label_dst, crop_size=640, raw_img_path=RAW_IMG_PATH, raw_label_path=RAW_LABEL_PATH):
    img = cv2.imread(raw_img_path+img_path)
    with open(raw_label_path+img_path[:-4]+'.txt') as f:
        label = [list(map(float, line.rstrip().split())) for line in f]
    label = np.array(label)
    img_h, img_w = img.shape[:2]
    resize_matrix = np.array([1, img_w, img_h, img_w, img_h])
    label_cropped = np.multiply(label, resize_matrix).astype(int)
    for i in range(35):
        x, y = i%5, i//5
        label_list = []
        crop_img = img[y*600:y*600+crop_size, x*600:x*600+crop_size]
        img_h, img_w = crop_img.shape[:2]
        shrink_matrix = np.array([1, 1/img_w, 1/img_h, 1/img_w, 1/img_h])
        cv2.imwrite(img_dst+name+'_{}_{}_{}.jpg'.format(crop_size, x, y), crop_img)
        for box in label_cropped:
            if x*600 < box[1] < x*600+640 and y*600 < box[2] < y*600+640:
                box_cropped = box
                box_cropped[1] = box[1]%600
                box_cropped[2] = box[2]%600
                box_cropped[3] = box[3]+5
                box_cropped[4] = box[4]+5
                box_cropped = np.round(np.multiply(box_cropped, shrink_matrix), 7)
                label_list.append(box_cropped.tolist())
        if label_list:
            with open(label_dst+name+'_{}_{}_{}.txt'.format(crop_size, x, y), 'w') as file:
                file.write('\n'.join(' '.join(str(content) for content in box) for box in label_list))

            
            
def move(file_list, img_dst, label_dst, img_src=RAW_IMG_PATH, label_src=RAW_LABEL_PATH, crop=False, crop_size=CROP_SIZE):
    print('Moving files ...')
    file_list=tqdm(file_list)
    if not crop:
        for each in file_list:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            shutil.move(img_src+each, img_dst+name+'.jpg')
            shutil.move(label_src+each[:-4]+'.txt', label_dst+name+'.txt')
    else:
        for path in file_list:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            crop_img(path, name, img_dst, label_dst)
    print('All files moved!')
                
                
def check_label(img_list, label_list, raw_img_path=RAW_IMG_PATH):
    print('Checking labels ...')
    li = [f for f in img_list if not f[:-4]+'.txt' in label_list]
    for each in li:
        shutil.rmtree(os.path.join(raw_img_path,each))
        print('The label of {} is not found, the image is removed!'.format(each))
    print('All labels checked!')
    

def add_background(img_dst, k=int(RATIO*BACKGROUND_NUMBER), background_path=BACKGROUND_PATH, crop=False, crop_size=CROP_SIZE):
    background_list = []
    for background in background_path:
        background_list = [f for f in os.listdir(background) 
                        if (os.path.isfile(os.path.join(background, f)) and f.endswith('.jpg'))]
        background_list = random.choices(background_list, k=k)
        for path in background_list:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            if crop:
                img = cv2.imread(background+path)
                crop_img = img[:crop_size, :crop_size]
                cv2.imwrite(img_dst+name+'_{}.jpg'.format(crop_size), crop_img)
            else:
                shutil.move(background+path, img_dst+name+'.jpg')
        

def train_test_split(raw_img_path=RAW_IMG_PATH, raw_label_path=RAW_LABEL_PATH, img_dst=IMG_DST, label_dst=LABEL_DST,
                     ratio=RATIO, train=TRAIN, test=TEST):
    img_list, label_list = get_lists(raw_img_path=raw_img_path, raw_label_path=raw_label_path)
    check_label(img_list, label_list)
    train_list = random.sample(img_list, int(ratio*len(img_list)))
    test_list = [item for item in img_list if item not in train_list]
    move(train_list, img_dst=img_dst+train, label_dst=label_dst+train, crop=False)
    move(test_list, img_dst=img_dst+test, label_dst=label_dst+test, crop=False)
    # add_background(img_dst=img_dst+train, k=int(RATIO*BACKGROUND_NUMBER))
    # add_background(img_dst=img_dst+test, k=int((1-RATIO)*BACKGROUND_NUMBER))
    
def main():
    train_test_split()

if __name__ == '__main__':
    main()