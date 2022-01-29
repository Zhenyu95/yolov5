# unzip and move files
mv /Users/zhenyu/Desktop/dataset_1230.zip /root/
mv /dataset_test_OK.z01 /root/
mv /dataset_test_OK.z02 /root/
mv /dataset_test_OK.z03 /root/
mv /dataset_test_OK.zip /root/
mv /OK_Images_496.zip /root/
mv /dataset_test_NG.zip /root/

cd ..
zip -FF dataset_test_OK.zip --out dataset_test_OK_full.zip
unzip dataset_test_OK_full.zip
unzip dataset_1230.zip
unzip OK_Images_496.zip
unzip dataset_test_NG.zip

rm dataset_1230/images/test/*
rm dataset_1230/labels/test/*

python3 /root/yolov5/random_move.py

mv dataset_test_OK/images/test/* dataset_1230/images/test/

mv dataset_test_NG/images/test/* dataset_1230/images/test/
mv dataset_test_NG/labels/test/* dataset_1230/labels/test/

mv 496/* ./dataset_1230/images/train/

cd yolov5
# python3 train.py --epochs 100 --data iphoneaoi.yaml --batch 32 --weights yolov5x.pt --img 1920 --cache --evolve 1000
python3 -m torch.distributed.launch --nproc_per_node 4 train.py --batch 24 --data iphoneaoi.yaml --weights yolov5x.pt --device 0,1,2,3 --epochs 300 --img 1920 --rect -adam
# for i in 0 1 2 3 4 5 6 7; do
#   nohup python3 train.py --epochs 50 --data iphoneaoi.yaml --weights yolov5x.pt --cache --evolve 500 --batch 32 --img 1920 --rect --device $i > evolve_gpu_$i.log &
# done
python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --img 1920 1440 --include coreml
# zip -r runs.zip runs/