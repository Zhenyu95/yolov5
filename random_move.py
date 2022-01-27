import random
import shutil
import os
from tqdm import tqdm

random.seed(9)
img_path = '/root/dataset_test_NG/images/test/'
label_path = '/root/dataset_test_NG/labels/test/'
img_dst = '/root/dataset_1230/images/train/'
label_dst = '/root/dataset_1230/labels/train/'

# img_path = '/Users/zhenyu/Desktop/dataset_test_NG/images/test/'
# label_path = '/Users/zhenyu/Desktop/dataset_test_NG/labels/test/'
# img_dst = '/Users/zhenyu/Desktop/untitled/'
# label_dst = '/Users/zhenyu/Desktop/untitled/'

file_list = [f[:-4] for f in os.listdir(img_path) if f.endswith('.jpg')]
rand_list = random.sample(file_list, 100)

for each in tqdm(rand_list):
    shutil.move(img_path+each+'.jpg', img_dst+each+'.jpg')
    shutil.move(label_path+each+'.txt', label_dst+each+'.txt')

print('-'*50+'files moved'+'-'*50)