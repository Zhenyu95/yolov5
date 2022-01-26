import random
import shutil
import os

random.seed(9)

file_list = [f[:-4] for f in os.listdir('/root/dataset_test_NG/images/test/') if f.endswith('.jpg')]
rand_list = random.sample(file_list, 100)

for each in rand_list:
    shutil.move('/root/dataset_test_NG/images/test/'+each+'.jpg', '/root/dataset_1230/images/train/'+each+'.jpg')
    shutil.move('/root/dataset_test_NG/labels/test/'+each+'.txt', '/root/dataset_1230/labels/train/'+each+'.txt')