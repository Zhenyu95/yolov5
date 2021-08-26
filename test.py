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
simcloud -c mr2 job post --cpus 16 --gpus 4 --memory 128 --attributes gpu_brand:Tesla --smi current-ubuntu18.04-cuda11.0 \
    --bundle bundle-7cc396102ef14675bb6b296c139dfecb --bundle bundle-d8bc71b0a93a4ed9b0df323edc9a6046 \
        --bundle bundle-8a9ca9927597454dbdcb0468dec80f57 --bundle bundle-1ca68869f5224788b1cac5e5a5b6900e  --ssh-login --timeout 335h

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get install software-properties-common
sudo apt-get update
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update