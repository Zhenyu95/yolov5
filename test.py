import os

os.system('sudo apt-get update')
os.system('sudo apt-get install git -y')
os.system('sudo apt-get install python3-pip -y')
os.system('pip3 install --upgrade pip')
os.system('sudo apt-get install zip unzip')
os.system('git clone https://github.com/Zhenyu95/yolov5')
os.system('sudo -H pip3 install --ignore-installed PyYAML')
os.system('pip3 install -r yolov5/requirements.txt')
# os.system('python3 yolov5/test.py')
# os.system('python3 yolov5/train.py --img 640 --batch 16 --epochs 3 --data yolov5/data/coco128.yaml --weights yolov5s.pt')
# print('*'*20, 'starting to zip output', '*'*20)
# os.system('zip -r archive.zip runs/train/')

# 
# simcloud job post --cpus 8 --gpus 2 --memory 128 --timeout 5m --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 --bundle bundle-70b32fe01f4841faa11f1802ca7af3d5 --command "python3 /Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/test_run.py" --output yolov5/models/yolov5s.yaml
simcloud -c mr2 job post --instance-type g8-large --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 --bundle bundle-9efee8ee9a054c078d43e2156445a208 --ssh-login --timeout 336h

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get install software-properties-common
sudo apt-get update
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update


python3 -m torch.distributed.launch --nproc_per_node 2 train.py --batch 24 --data iphoneaoi.yaml --weights yolov5l6.pt --device 0,1 --epochs 60 --img 3072 --rect

python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --img 3072 2304 --include coreml

simcloud -c mr2 job post --instance-type g8-large --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 --bundle bundle-2be480f911374c16bd8ed2909a19359d --bundle bundle-8a30068f1a0141e3ba3abe3fafecb61c --bundle bundle-f62419072b3648b59096cc34a10f7e99 --bundle bundle-2ba74e9b3f8c46ed943194577ac1854c --bundle bundle-de0ca781b2fe45b29a33e58988bf0b18 --bundle bundle-952ec6b48ca7492a881f403ac702999a --command "sh /Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/packages.sh" --output /root/yolov5/runs/ --output-to-bundle --timeout 336h