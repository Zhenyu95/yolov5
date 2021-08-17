sudo apt update
sudo apt install build-essential
sudo apt-get install unzip
sudo apt-get install git -y

# install pip
sudo apt-get -y install python3-pip
pip3 install --upgrade pip
python3 -m ensurepip --upgrade

git clone https://github.com/Zhenyu95/yolov5

python3 yolov5/run.py

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

# # TensorFlow
# pip3 install tensorflow

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

# extras --------------------------------------
# Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
# pycocotools>=2.0  # COCO mAP
# albumentations>=1.0.3