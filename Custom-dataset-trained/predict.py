from ultralytics import YOLO

# Load a pretrained YOLO11n-seg Segment model
model = YOLO(r"C:\Users\Khushi.Jirge\sem-seg\last.pt")

# Run inference on an image
results = model(r"C:\Users\Khushi.Jirge\sem-seg\2024-09-20 15.24.png")  # results list

# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks