# unzip and move files
mv /Users/zhenyu/Desktop/dataset_1230.zip /root/
cd ..
unzip dataset_1230.zip
cd yolov5
python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 32 --data iphoneaoi.yaml --weights yolov5x.pt --device 0,1,2,3,4,5,6,7 --epochs 30 --img 1920 --rect
for i in 0 1 2 3 4 5 6 7; do
  nohup python3 train.py --epochs 10 --data iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --cache --evolve --device $i > evolve_gpu_$i.log &
done
python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --img 1920 1440 --include coreml