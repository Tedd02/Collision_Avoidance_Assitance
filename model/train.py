import os
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # build a new model from scratch

# Use the model
names = model.names
print(names)
results = model.train(data="config.yaml", epochs=10)  # train the model
#results =  model.val()