import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# 1. CONFIGURATION
IMAGE_PATH = "test_image.jpg" 

# Path to your optimized INT8 model
MODEL_PATH = "student_int8_qat_optimized.onnx"
CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]

IMG_SIZE = 224
NUM_RUNS = 100

# 2. HELPER FUNCTIONS
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def preprocess_image(image_path):
    """
    Loads and preprocesses image to match training transforms:
    Resize -> ToTensor -> Normalize (ImageNet stats)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).numpy()
    return input_tensor

# 3. MAIN PREDICTION LOOP
def main():
    print(f"Loading model: {MODEL_PATH}...")
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"Processing image: {IMAGE_PATH}...")
    try:
        input_data = preprocess_image(IMAGE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    print("Warming up engine...")
    for _ in range(10):
        sess.run([output_name], {input_name: input_data})

    print(f"Running inference ({NUM_RUNS} loops for benchmarking)...")
    start = time.time()
    for _ in range(NUM_RUNS):
        logits = sess.run([output_name], {input_name: input_data})[0]
    end = time.time()

    avg_time_sec = (end - start) / NUM_RUNS
    avg_time_ms = avg_time_sec * 1000
    fps = 1 / avg_time_sec

    probs = softmax(logits)[0]
    pred_id = probs.argmax()
    pred_class = CLASSES[pred_id] if pred_id < len(CLASSES) else f"Unknown ID {pred_id}"
    confidence = probs[pred_id] * 100


    # 4. FINAL OUTPUT
    print("\n" + "="*40)
    print(f"🌱 PREDICTION RESULTS")
    print("="*40)
    print(f"Predicted Class:    {pred_class.upper()}")
    print(f"Confidence:         {confidence:.2f}%")
    print("-" * 40)
    print(f"Inference Time:     {avg_time_ms:.2f} ms")
    print(f"Throughput:         {fps:.2f} FPS")
    print("="*40)

if __name__ == "__main__":
    main()
