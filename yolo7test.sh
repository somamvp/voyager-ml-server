git clone https://github.com/WongKinYiu/yolov7
cd yolov7

# pip install -qr requirements.txt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg