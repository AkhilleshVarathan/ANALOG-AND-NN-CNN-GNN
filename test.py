import torch
import os
import random
import cv2

# Path to local YOLOv5 repo
local_yolov5_path = r"C:\Users\Deepak Skandh\yolov5"

# Path to your trained model
model_path = os.path.join(local_yolov5_path, 'runs', 'train', 'obstacle_detector_v1', 'weights', 'best.pt')

# Load custom trained YOLOv5 model
model = torch.hub.load(local_yolov5_path, 'custom', path=model_path, source='local')

model.eval()

# Path to test images
val_folder = r"D:\object_detection_3\blind2.v2i.yolov5pytorch\test\images"
image_extensions = ['.jpg', '.jpeg', '.png']
image_paths = [os.path.join(val_folder, f) for f in os.listdir(val_folder)
               if os.path.splitext(f)[1].lower() in image_extensions]

# Randomly pick 10 images
sampled_images = random.sample(image_paths, 10)

# Inference and draw boxes
for img_path in sampled_images:
    results = model(img_path)
    preds = results.xyxy[0]  # x1, y1, x2, y2, conf, class
    img = cv2.imread(img_path)

    for *box, conf, cls in preds:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show image
    cv2.imshow("YOLOv5 Custom", img)
    cv2.waitKey(0)

    # Optional: Save the image
    save_dir = os.path.join("runs", "detect", "custom_test")
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

cv2.destroyAllWindows()
