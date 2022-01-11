sudo apt update
sudo apt install build-essential
sudo apt-get install unzip
sudo apt-get install git -y

# install yolov5 packages
sudo apt-get -y install python3-pip
pip3 install --upgrade pip
python3 -m ensurepip --upgrade
sudo -H pip3 install --ignore-installed PyYAML

git clone https://github.com/Zhenyu95/yolov5
pip3 install -r yolov5/requirements.txt

cd yolov5
sh training.sh

# # base ----------------------------------------

# pip3 install matplotlib>=3.2.2
# pip3 install numpy>=1.18.5
# pip3 install opencv-python>=4.1.2
# pip3 install Pillow
# pip3 install PyYAML>=5.3.1
# pip3 install scipy>=1.4.1


# # PyTorch

# pip3 install torch>=1.7.0
# pip3 install torchvision>=0.8.1
# pip3 install tqdm>=4.41.0


# # logging -------------------------------------

# pip3 install tensorboard>=2.4.1
# # wandb


# # plotting ------------------------------------

# pip3 install seaborn>=0.11.0
# pip3 install pandas


# export --------------------------------------

# coremltools>=4.1
# onnx>=1.9.0
# scikit-learn==0.19.2  # for coreml quantization


# following packages are for tensorflow multi-GPU

# pip3 install tensorflow==2.4.0
# sudo dpkg -i libcudnn8_8.0.2.39-1+cuda11.0_amd64.deb
# sudo dpkg -i libcudnn8-dev_8.0.2.39-1+cuda11.0_amd64.deb
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo apt-get install software-properties-common -y
# sudo apt-get update
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# sudo apt-get update