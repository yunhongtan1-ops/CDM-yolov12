
 
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v12/yolov12_SimAM.yaml") 

# Train the model
results = model.train(data="iSSOD.yaml", epochs=200, imgsz=1024)