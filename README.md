# Lightweight-and-Efficient-CNN

## Introduction
As the world accelerates towards **"Physical AI"**, the demand for deploying intelligent systems on edge devices (drones, mobile phones, IoT sensors) is exploding. These environments demand models that are not just accurate, but also **lightweight (small memory footprint)** and **efficient (fast inference)**.

This repository hosts a robust framework for building such models using a synergistic combination of **Knowledge Distillation (KD)** and **Quantization Aware Training (QAT)**. 

The goal is simple: Create models that mirror the high accuracy of massive Deep Learning architectures while remaining small enough to run on constrained hardware.

---

## Datasets Implemented
This repository contains the full implementation of the framework on two specific datasets:

1.  **[Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset):** 5 varieties of rice (Arborio, Basmati, Ipsala, Jasmine, Karacadag).
2.  **[Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset/data):** Classification of rice plant diseases.

> I have also applied a **novel, advanced version** of this methodology to standard benchmarks (**CIFAR-10** and **CIFAR-100**), achieving SOTA results.
> * For detailed results and analysis on these benchmarks, please visit the directory: `Additional-Benchmark-Datasets`.

---

##  Methodology Framework

Our pipeline ensures maximum efficiency without sacrificing predictive power:

1.  **Dataset Analysis:** We begin by analyzing variance and outliers to determine model suitability.
2.  **Teacher Selection:** We select an ensemble of 3 Wide, Deep, and Heavy **Teacher Models** (SOTA architectures) that demonstrate excellent generalization on the specific dataset.
3.  **Student Selection:** We identify a **Student Model** that balances learning capacity with quantization robustness (ensuring it doesn't collapse when precision is reduced).
4.  **SOTA Augmentation:** All models are trained using advanced augmentation techniques tailored to the dataset to maximize baseline accuracy.
5.  **Ensemble Distillation:** We perform **Knowledge Distillation** by averaging logits from the Teacher Ensemble. This transfers "dark knowledge," allowing the Student to push past the accuracy limits of standard "from-scratch" training.
6.  **Quantization Aware Training (QAT):** The Student model (FP32) undergoes QAT. We use a combined loss function (`CrossEntropy Loss + KD Loss`) during the backward pass while "fake-converting" the model to INT8 to simulate quantization noise.
7.  **Edge Deployment:** The final model is converted to **ONNX format** for optimized inference on edge devices.



---

## Results

### 1. Rice Image Dataset
* **Teacher Ensemble:** ResNet18 (99.87%), MobileNetV2 (99.89%), DenseNet121 (99.91%)
* **Student Model:** MobileNetV3-Small (Width 0.25x)

| Metric | FP32 Baseline | INT8 Quantized | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | 0.45 MB | **0.31 MB** | **~31% Reduction** |
| **Accuracy** | 99.66% | **98.07%** | **High retention** |
| **Inference Latency** | 5.66 ms | **3.95 ms** | **~1.4x Faster** |

*(Hardware: Kaggle Notebook CPU - Intel Xeon @ 2.20GHz)*

### 2. Rice Disease Dataset
* **Teacher Ensemble:** ResNet18 (97.65%), MobileNetV2 (97.65%), DenseNet121 (98.17%)
* **Student Model:** MobileNetV3-Small (Width 0.50x)

| Metric | FP32 Baseline | INT8 Quantized | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | 3.32 MB | **1.10 MB** | **3.0x Smaller** |
| **Accuracy** | 97.29% | 91.25% | -6.04% Diff |
| **Inference Latency** | 5.59 ms | **1.89 ms** | **~3.0x Faster** |
| **Throughput** | 178.8 FPS | **529.5 FPS** | **~3.0x Higher** |

*(Hardware: Kaggle Notebook CPU - Intel Xeon @ 2.20GHz)*

---

##  Repository Structure

* `Additional-Benchmark-Datasets/` - **Highly Recommended:** Contains information regarding the novel SOTA methodology applied to CIFAR-10/100.
* `Mobile-App-Demo/` - Contains demo videos of a basic mobile app created to deploy these INT8 ONNX models directly on a device.
* `Model_files/` - Stores the trained `.onnx` files and model weights.
* `Prediction/` - Contains `predict.py` for users to easily test the models on new data.
* `Train-Notebooks/` - Complete training code for the entire framework (Teachers, Distillation, QAT).

---

##  Conclusion
This framework provides a highly optimized pathway for developers and researchers pursuing **Physical AI**. By leveraging the dual power of Ensemble Distillation and Quantization, we prove that you do not need massive hardware to run intelligent visual recognition systems.
