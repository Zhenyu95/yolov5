# move zip files to /root
mv /Synthesized.z01 /root/
mv /Synthesized.z02 /root/
mv /Synthesized.z03 /root/
mv /Synthesized.z04 /root/
mv /Synthesized.z05 /root/
mv /Synthesized.z06 /root/
mv /Synthesized.zip /root/
mv /OK.z01 /root/
mv /OK.zip /root/
mv /Real.zip /root/

# unzip zip files
cd /root
zip -FF Synthesized.zip --out Synthesized_full.zip
zip -FF OK.zip --out OK_full.zip
unzip Synthesized_full.zip
unzip OK_full.zip
unzip Real.zip

# move FOV1-4 images together
mv /root/OK/FOV1/* /root/OK/
mv /root/OK/FOV2/* /root/OK/
mv /root/OK/FOV3/* /root/OK/
mv /root/OK/FOV4/* /root/OK/
rm -r /root/OK/FOV1/
rm -r /root/OK/FOV2/
rm -r /root/OK/FOV3/
rm -r /root/OK/FOV4/

# move classes.txt to training labels path
# rm /root/Synthesized/labels/classes.txt
mv /root/Real/labels/classes.txt /root/dataset/labels/train/

# move revised label to update Real labels
# mv /labels.zip /root/
# unzip labels.zip
# mv /root/labels/* /root/Real/labels/

# create dataset folders based on yolov5 structure requirement
mkdir /root/dataset/
mkdir /root/dataset/images/
mkdir /root/dataset/labels/
mkdir /root/dataset/images/train/
mkdir /root/dataset/images/test/
mkdir /root/dataset/labels/train/
mkdir /root/dataset/labels/test/

# random train/test split, ratio is defined in random_move.py
python3 /root/yolov5/random_move.py

cd yolov5
# python3 train.py --epochs 1000 --data iphoneaoi.yaml --batch 16 --weights yolov5x.pt --img 1920 --cache --evolve 1000 --rect --adam
# python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 128 --data iphoneaoi.yaml --weights yolov5s.pt --device 0,1,2,3,4,5,6,7 --epochs 600 --img 1920 --adam --cache
python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 32 --data iphoneaoi.yaml --weights yolov5x.pt --device 0,1,2,3,4,5,6,7 --epochs 200 --img 1280 --adam --cache
# python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 128 --data iphoneaoi.yaml --cfg models/hub/yolov5s-transformer.yaml --device 0,1,2,3,4,5,6,7 --epochs 600 --img 1920 --rect --adam --cache
# python3 -m torch.distributed.launch --nproc_per_node 8 train.py --batch 32 --data iphoneaoi.yaml --cfg models/hub/yolov5x-transformer.yaml --device 0,1,2,3,4,5,6,7 --epochs 600 --img 1920 --rect --adam --cache
# for i in 0 1 2 3 4 5 6 7; do
#   nohup python3 train.py --epochs 50 --data iphoneaoi.yaml --weights yolov5x.pt --cache --evolve 500 --batch 32 --img 1920 --rect --device $i > evolve_gpu_$i.log &
# done
python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp/weights/best.pt --img 1280 960 --include coreml
# python3 export.py --data data/iphoneaoi.yaml --weights runs/train/exp2/weights/best.pt --img 1920 1440 --include coreml
# zip -r runs.zip runs/