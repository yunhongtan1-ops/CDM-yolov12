
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v12/yolov12-CA-DySample-MSMF.yaml")
results = model.train(data="iSSOD.yaml", epochs=300,batch=32)