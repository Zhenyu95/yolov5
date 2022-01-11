# unzip and move files
mv /Users/zhenyu/Desktop/dataset_1230.zip /root/
cd ..
unzip dataset_1230.zip
cd yolov5
python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 32 --data iphoneaoi.yaml --weights yolov5x.pt --device 0,1,2,3,4,5,6,7 --epochs 2 --img 1920 --rect