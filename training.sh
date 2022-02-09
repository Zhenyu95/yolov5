# unzip and move files
mv /dataset.z01 /root/
mv /dataset.z02 /root/
mv /dataset.z03 /root/
mv /dataset.z04 /root/
mv /dataset.zip /root/

cd ..
zip -FF dataset.zip --out dataset_full.zip
unzip dataset_full.zip

cd yolov5
# python3 train.py --epochs 100 --data iphoneaoi.yaml --batch 32 --weights yolov5x.pt --img 1920 --cache --evolve 1000
python3 -m torch.distributed.launch --nproc_per_node 4 train.py --batch 24 --data iphoneaoi.yaml --weights yolov5x.pt --device 0,1,2,3 --epochs 300 --img 1920 --rect
# for i in 0 1 2 3 4 5 6 7; do
#   nohup python3 train.py --epochs 50 --data iphoneaoi.yaml --weights yolov5x.pt --cache --evolve 500 --batch 32 --img 1920 --rect --device $i > evolve_gpu_$i.log &
# done
python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --img 1920 1440 --include coreml
# zip -r runs.zip runs/