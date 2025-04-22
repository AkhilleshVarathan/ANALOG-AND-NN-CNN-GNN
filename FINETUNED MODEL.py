import os
import sys
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
import torch.nn as nn


# 🔧 Paths
yolov5_path = r"C:\Users\Deepak Skandh\yolov5"
sys.path.append(yolov5_path)
from models.yolo import Model


# 📦 Import scripts
from train import main as train_yolo
from val import main as val_yolo
from detect import run as detect_yolo

# 🔨 Options
opt = {
    'weights': os.path.join(yolov5_path, 'yolov5s.pt'),
    'cfg': os.path.join(yolov5_path, 'models', 'yolov5s.yaml'),
    'data': r"D:\object_detection_3\blind2.v2i.yolov5pytorch\data.yaml",

    'epochs': 40,
    'batch_size': 32,
    'imgsz': 640,
    'rect': False,
    'resume': False,
    'image_weights': False,
    'freeze': [x for x in range(8)],

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # ✅ Force GPU if available
    'workers': min(8, os.cpu_count() - 1),

    'project': os.path.join(yolov5_path, 'runs', 'train'),
    'name': 'obstacle_detector_v1',
    'exist_ok': True,

    'hyp': os.path.join(yolov5_path, 'data', 'hyps', 'hyp.scratch-low.yaml'),
    'adam': False,
    'single_cls': False,
    'sync_bn': False,
    'optimizer': 'SGD',

    'noplots': False,
    'noval': False,
    'nosave': False,
    'noautoanchor': False,

    'multi_scale': False,
    'label_smoothing': 0.0,
    'patience': 15,
    'quad': False,
    'cos_lr': False,
    'save_period': -1,
    'evolve': False,

    # Missing CLI defaults
    'cache': None,
    'bbox_interval': -1,
    'close_mosaic': 0,
    'upload_dataset': False,
    'entity': None,
    'local_rank': -1,
    'save_dir': None,
    'seed': 0,
    'verbose': False,
    'save_json': False,
    'save_hybrid': False,
    'linear_lr': False,
    'prefix': '',
    'amp': True,  # ✅ AMP for mixed precision training (optional but great!)
}

opt = SimpleNamespace(**opt)

# ✅ Select device
device = torch.device(opt.device)
print(f"🚀 Using device: {device}")
if device.type == 'cuda':
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"🧠 Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"📦 Memory Reserved:  {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

# ✅ Load YOLOv5 model and move to GPU
model = Model(opt.cfg).to(device)


# Freeze 90% of the backbone layers (Layer 0 to Layer 6)
backbone_layers = [model.model[i] for i in range(7)]  # Layers 0-6
for layer in backbone_layers:
    for param in layer.parameters():
        param.requires_grad = False

# Freeze 50% of the neck layers (Layer 9 to Layer 14)
neck_layers = [model.model[i] for i in range(9, 15)]  # Layers 9-14
for layer in neck_layers:
    for param in layer.parameters():
        param.requires_grad = False

# Head layers (Layer 15 onwards) will remain trainable by default, no need to modify them

# Track performance metrics and time duration
epoch_metrics = {'train_loss': [], 'val_loss': [], 'train_ap': [], 'val_ap': []}
epoch_times = []

# 🚀 Train + Validate + Detect
if __name__ == '__main__':
    try:
        assert Path(opt.weights).exists(), f"Weights missing: {opt.weights}"
        assert Path(opt.data).exists(), f"Data YAML missing: {opt.data}"
        assert Path(opt.cfg).exists(), f"Model config missing: {opt.cfg}"

        print(f"🚀 Starting training (Torch {torch.__version__}) on device {opt.device}")

        # Start Training
        for epoch in range(opt.epochs):
            start_time = time.time()  # Track time for each epoch

            print(f"\n🔄 Epoch {epoch + 1}/{opt.epochs}")

            # Training the model
            train_metrics = train_yolo(opt)

            # Store train metrics for plotting later (e.g., train_loss, train_ap)
            epoch_metrics['train_loss'].append(train_metrics['loss'])
            epoch_metrics['train_ap'].append(train_metrics['ap'])

            # Track epoch duration
            epoch_duration = time.time() - start_time
            epoch_times.append(epoch_duration)
            print(f"⏱️ Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

            # Validation (optional, can be commented out if not needed for each epoch)
            if not opt.noval:
                print(f"\n✅ Validation on validation set (Epoch {epoch + 1}/{opt.epochs})...")
                val_metrics = val_yolo(opt)

                # Store validation metrics for plotting later (e.g., val_loss, val_ap)
                epoch_metrics['val_loss'].append(val_metrics['loss'])
                epoch_metrics['val_ap'].append(val_metrics['ap'])

        print(f"\n✅ Finished training, now running detection")

        # Running detection on test images
        detect_yolo(
            weights=os.path.join(
                yolov5_path, 'runs', 'train', opt.name, 'weights', 'best.pt'),
            source=r"D:\object_detection_3\blind2.v2i.yolov5pytorch\test",
            imgsz=640,
            conf_thres=0.25,
            iou_thres=0.45,
            save_txt=False,
            save_conf=True,
            save_crop=False,
            project=os.path.join(yolov5_path, 'runs', 'detect'),
            name='obstacle_detector_v1_test',
            exist_ok=True,
            line_thickness=2
        )

        print(f"\n📁 Detection results saved to 'runs/detect/obstacle_detector_v1_test'")

        # Generate graphs at the end of training
        print("\n📊 Generating performance graphs...")

        # Plotting loss and AP curves for both train and validation
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Training and Validation Loss
        ax[0].plot(range(1, opt.epochs + 1), epoch_metrics['train_loss'], label='Train Loss')
        ax[0].plot(range(1, opt.epochs + 1), epoch_metrics['val_loss'], label='Validation Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].legend()

        # Plot Training and Validation AP (Average Precision)
        ax[1].plot(range(1, opt.epochs + 1), epoch_metrics['train_ap'], label='Train AP')
        ax[1].plot(range(1, opt.epochs + 1), epoch_metrics['val_ap'], label='Validation AP')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Average Precision')
        ax[1].set_title('Training and Validation Average Precision (AP)')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        # Plotting epoch duration
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, opt.epochs + 1), epoch_times, label='Epoch Duration (seconds)', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Time (seconds)')
        plt.title('Epoch Duration Over Training')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\n❌ Training failed: {type(e).__name__}: {str(e)}")
        if "has no attribute" in str(e):
            missing = str(e).split("'")[1]
            print(f"⚠️ Solution: Add '{missing}': False or appropriate value to your config")

        print("\n🔧 Quick fixes:")
        print("- Missing params? Compare with YOLOv5 CLI args")
        print("- CUDA errors? Try device='cpu'") 
        print("- Path issues? Run: print(Path(opt.data).exists())")
