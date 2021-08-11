import os

os.system('sudo apt-get update')
os.system('sudo apt-get install git -y')
os.system('sudo apt-get install python3-pip -y')
os.system('pip3 install --upgrade pip')
os.system('sudo apt-get install zip unzip')
os.system('git clone https://github.com/Zhenyu95/yolov5')
os.system('sudo -H pip3 install --ignore-installed PyYAML')
os.system('pip3 install -r yolov5/requirements.txt')
# os.system('python3 yolov5/train.py --img 640 --batch 16 --epochs 3 --data yolov5/data/coco128.yaml --weights yolov5s.pt')
# print('*'*20, 'starting to zip output', '*'*20)
# os.system('zip -r archive.zip runs/train/')

# 
# simcloud job post --cpus 8 --gpus 2 --memory 128 --timeout 5m --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 --bundle bundle-70b32fe01f4841faa11f1802ca7af3d5 --command "python3 /Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/test_run.py" --output yolov5/models/yolov5s.yaml
# simcloud job post --cpus 8 --gpus 2 --memory 128 --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 --bundle bundle-c59ffba836c5491aa3a58b45b1836f27 --ssh-login