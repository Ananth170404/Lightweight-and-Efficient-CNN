# Unified Knowledge Distillation and Mixed-Precision Quantization for Efficient CNNs

## Introduction
This repository presents a novel, synergistic methodology designed to create highly lightweight and efficient Convolutional Neural Networks (CNNs). By bringing together the worlds of **Knowledge Distillation (KD)** and **Quantization**, this approach achieves state-of-the-art performance in model compression without compromising accuracy.

The core objective is to democratize deep learning deployment on edge devices by significantly reducing memory footprint and computational cost while mirroring the predictive power of heavy, over-parameterized models.

## Datasets
The methodology has been rigorously benchmarked against standard academic datasets:
* **CIFAR-10**
* **CIFAR-100**

## Methodology
Our approach goes beyond standard quantization techniques. We employ a multi-stage pipeline designed to maximize information retention during the compression process:

1.  **State-of-the-Art Augmentation:** Advanced data augmentation techniques are utilized to train all baselines and ensembles.
2.  **Teacher Ensemble Construction:** We specifically select wide and dense teacher models (e.g., ResNeXt, DenseNet) to form a robust knowledge base.
3.  **Student Selection:** Identification of "Good Quantizable" student architectures (e.g., MobileNetV2, ResNet18) that are resilient to precision reduction.
4.  **Novel Dynamic Knowledge Transfer:** Instead of standard averaging or majority voting, we utilize a **novel dynamic distillation method** to transfer knowledge from the ensemble to the student.
5.  **Sensitivity Analysis (Layer-wise):** For mixed-precision quantization, we perform a layer-wise sensitivity analysis to assign bit-widths intelligently. Critical layers retain higher precision, while robust layers are quantized aggressively.
6.  **Quantization-Aware Distillation:** We implement Knowledge Distillation *during* the quantization process (QAT) to convert models to Uniform 16-bit, 8-bit, or Mixed 16/8-bit and 8/4-bit schemes.

> ** Note on Code Availability**
>
> This methodology represents a significant advancement over other implementations found in this repository. It is currently **State-Of-The-Art (SOTA)** in the Efficient-CNN domain.
>
> **I am currently authoring a research paper on this methodology for publication in an international conference.** Consequently, the source code for these specific experiments is **not yet public**. It will be released upon the acceptance and publication of the paper.

## Results of Experiments

### 1. CIFAR-10 Benchmark
* **Teacher Ensemble:** DenseNet 190_40 (97.03%), ResNet 28_10 (97.24%), ResNet 28_4 (96.66%)
* **Student Model:** MobileNetV2

| Configuration | Bit-Width Scheme | Accuracy |
| :--- | :--- | :--- |
| **Baseline (Scratch)** | FP32 | 94.46% |
| **KD (FP32)** | FP32 | **96.34%** |
| QAT Scratch | 16-bit | 94.54% |
| QAT KD | 16-bit | 95.20% |
| QAT Scratch | 8-bit | 94.48% |
| QAT KD | 8-bit | 94.81% |
| QAT Layer+KD | Mixed 16/8-bit | **96.40%** |
| QAT Layer+KD | Mixed 8/4-bit | **95.89%** |

### 2. CIFAR-100 Benchmark
* **Teacher Ensemble:** ResNeXt 29_8x64d (83.64%), ResNet 28_10 (84.39%), DenseNet 121 (83.95%)
* **Student Model:** ResNet18

| Configuration | Bit-Width Scheme | Accuracy |
| :--- | :--- | :--- |
| **Baseline (Scratch)** | FP32 | 81.11% |
| **KD (FP32)** | FP32 | **83.66%** |
| QAT Scratch | 16-bit | 78.96% |
| QAT KD | 16-bit | 83.47% |
| QAT Scratch | 8-bit | 79.01% |
| QAT KD | 8-bit | 83.73% |
| QAT Layer+KD | Mixed 16/8-bit | **83.59%** |
| QAT Layer+KD | Mixed 8/4-bit | **83.04%** |

## Efficiency Analysis
The primary success of this project is maintaining near-teacher accuracy while drastically reducing the model size.

### MobileNetV2 (CIFAR-10)
* **Original Size (FP32):** 9.2 MB
* **Compressed Size (Mixed 8/4):** 1.38 MB
* **Reduction:** **~6.6x smaller** with higher accuracy than the scratch FP32 model.

### ResNet18 (CIFAR-100)
* **Original Size (FP32):** 44.8 MB
* **Compressed Size (Mixed 8/4):** 6.72 MB
* **Reduction:** **~6.6x smaller** while outperforming the scratch FP32 model.

## Inferences
1.  **Augmentation Impact:** State-of-the-art augmentation heavily boosted the accuracy of all models across the board.
2.  **Ensemble Superiority:** The usage of the novel ensemble-based dynamic distillation proved significantly better than general averaging or majority-voting methods.
3.  **Accuracy Retention:** Despite transferring the knowledge to a much lighter student model and even quantizing it down to Mixed-8/4 bit schemes, the accuracy mirrors the heavier teacher models. This is largely attributed to the proprietary layer analysis method used to boost accuracy.

## Detailed Data
For a granular view of the experiments, including precise configurations and raw numbers, please refer to the Excel file located at:

`Additional-Benchmark-Datasets/Results of Experiments- IITM.xlsx`
