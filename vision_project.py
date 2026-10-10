"""Computer Vision project: Keras CNN classifier + YOLOv8 object detection with OpenCV.

Part 1  Train a CNN (TensorFlow/Keras) on CIFAR-10 and report test accuracy.
Part 2  Run a pretrained YOLOv8n detector on your own images (OpenCV for I/O + drawing),
        and report detections and average inference time per image.

Usage:
    pip install -r requirements.txt
    python vision_project.py --images my_images/        # folder with a few .jpg/.png files
Every number is measured on your machine and saved to outputs/results.json.
"""
import argparse, glob, json, os, time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO

SEED = 42
tf.keras.utils.set_random_seed(SEED)
os.makedirs("outputs", exist_ok=True)
results = {}

# ---------- Part 1: Keras CNN on CIFAR-10 ----------
(x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
x_tr, x_te = x_tr.astype("float32") / 255.0, x_te.astype("float32") / 255.0

model = keras.Sequential([
    keras.layers.Input((32, 32, 3)),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(x_tr, y_tr, validation_split=0.1, epochs=15, batch_size=128, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
loss, acc = model.evaluate(x_te, y_te, verbose=0)
model.save("outputs/cifar10_cnn.keras")
results["cnn"] = {"dataset": "CIFAR-10", "train_images": 50000, "test_images": 10000,
                  "epochs_run": len(hist.history["loss"]), "test_accuracy": round(float(acc), 4),
                  "parameters": int(model.count_params())}
print("CNN test accuracy:", results["cnn"]["test_accuracy"])

# ---------- Part 2: YOLOv8 detection with OpenCV ----------
ap = argparse.ArgumentParser(); ap.add_argument("--images", default="images")
paths = sorted(p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(ap.parse_args().images, e)))
if paths:
    det = YOLO("yolov8n.pt")
    times, counts, per_class = [], [], {}
    for p in paths:
        img = cv2.imread(p)
        t0 = time.perf_counter(); r = det(img, verbose=False)[0]; times.append((time.perf_counter() - t0) * 1000)
        counts.append(len(r.boxes))
        for b in r.boxes:
            name = r.names[int(b.cls)]; per_class[name] = per_class.get(name, 0) + 1
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, f"{name} {float(b.conf):.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.imwrite(os.path.join("outputs", "det_" + os.path.basename(p)), img)
    times = times[1:] or times  # drop the first (warm-up) call
    results["yolo"] = {"model": "yolov8n (pretrained COCO)", "images": len(paths), "total_detections": int(sum(counts)),
                       "avg_inference_ms": round(float(np.mean(times)), 1), "detections_by_class": per_class}
    print("YOLO:", results["yolo"])
else:
    print("No images found - put some .jpg/.png files in the folder and pass --images <folder> to run Part 2.")

json.dump(results, open("outputs/results.json", "w"), indent=2)
