import onnxruntime as ort
import numpy as np
import time
from PIL import Image
from torchvision import transforms

# CONFIG
MODEL_PATH = MODEL_PATH["student_int8_static.onnx"]
IMAGE_PATH = IMAGE_PATH[".jpg"]

CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = 224
NUM_RUNS = 100    

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# PREPROCESS
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0).numpy() 

# ONNX RUNTIME
print("Loading ONNX Runtime session...")
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# WARM-UP
for _ in range(10):
    sess.run([output_name], {input_name: x})

# TIMING
start = time.time()
for _ in range(NUM_RUNS):
    logits = sess.run([output_name], {input_name: x})[0]
end = time.time()

avg_time_sec = (end - start) / NUM_RUNS
avg_time_ms = avg_time_sec * 1000

# PREDICTION
probs = softmax(logits)[0]
pred_id = probs.argmax()
pred_class = CLASSES[pred_id]
confidence = probs[pred_id] * 100

# OUTPUT
print("Prediction:", pred_class)
print(f"Confidence: {confidence:.2f}%")
print(f"Average inference time: {avg_time_ms:.2f} ms ({avg_time_sec:.4f} s)")
