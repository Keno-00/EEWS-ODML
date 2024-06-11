from ultralytics import YOLO

# Load a model

model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)
datapath = "D:/source/repos/EEWS-OD-ML/eews-odml-pre.v16i.yolov8/data.yaml"


# Train the model
results = model.train(data=datapath, epochs=3, imgsz=320)