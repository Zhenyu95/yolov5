import random
import shutil
import os
from tqdm import tqdm
import pandas as pd

random.seed(9)
syn_path = '/root/Synthesized/'
ok_path = '/root/OK/'
real_path = '/root/Real/'
dst = '/root/dataset/'

# move OK images
def move_OK(path, dst, n_train, n_val='all'):
    train_dst = os.path.join(dst, 'images', 'train')
    test_dst = os.path.join(dst, 'images', 'test')
    file_list = [f for f in os.listdir(path) if f.endswith('.jpg')]
    train_list = random.sample(file_list, n_train)
    
    
    # all the left files are test file
    test_list = [f for f in file_list if f not in train_list]
    if n_val != 'all':
        # randomly select certain number of files as test files
        test_list = random.sample(test_list, n_val)

    for file in tqdm(file_list):
        if file in train_list:
            shutil.move(os.path.join(path, file), os.path.join(train_dst, file))
        elif file in test_list:
            shutil.move(os.path.join(path, file), os.path.join(test_dst, file))
            
    print('-' * 50 + 'OK files moved' + '-' * 50)


def move(path, dst, n_train, n_val='all'):
    img_path = os.path.join(path, 'images')
    label_path = os.path.join(path, 'labels')
    train_img_dst = os.path.join(dst, 'images', 'train')
    train_label_dst = os.path.join(dst, 'labels', 'train')
    test_img_dst = os.path.join(dst, 'images', 'test')
    test_label_dst = os.path.join(dst, 'labels', 'test')

    img_list = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
    label_list = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    train_img_list = random.sample(img_list, n_train)
    train_label_list = random.sample(label_list, n_train)
    
    # all the left files are test file
    test_img_list = [f for f in img_list if f not in train_img_list]
    test_label_list = [f for f in label_list if f not in train_label_list]
    if n_val != 'all':
        # randomly select certain number of files as test files
        test_img_list = random.sample(test_img_list, n_val)
        test_label_list = random.sample(test_label_list, n_val)
    
    

    # sanity check
    null_img_list = [f for f in img_list if f[:-4]+'.txt' not in label_list]
    
    if null_img_list:
        raise FileNotFoundError('{} does not have label'.format(null_img_list))

    for img in tqdm(img_list):
        label = img[:-4] + '.txt'
        if img in train_img_list:
            shutil.move(os.path.join(img_path, img), os.path.join(train_img_dst, img))
            shutil.move(os.path.join(label_path, label), os.path.join(train_label_dst, label))
        elif img in test_img_list:
            shutil.move(os.path.join(img_path, img), os.path.join(test_img_dst, img))
            shutil.move(os.path.join(label_path, label), os.path.join(test_label_dst, label))
            
    print('-' * 50 + str(path) +'moved' + '-' * 50)
    
    df = pd.DataFrame(test_img_list)
    df.to_csv('/root/yolov5/real_test.csv')
    
    
    
move_OK(ok_path, dst, 3500)
move(real_path, dst, 180)
move(syn_path, dst, 8820, n_val=300)
