# Driver Inattention Classifier

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
[![Notebook](https://img.shields.io/badge/Kaggle-Notebook-orange.svg)](https://www.kaggle.com/code/saadaswad/cv-miniproject)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time driver monitoring system using MobileNetV3. Edge-optimized for 0.6ms latency and 1.96MB size, achieving >90% accuracy in classifying inattentive states.

## 🚀 Methodology & Evolution
The development of this system followed an iterative approach, documented through multiple notebook versions. We have selected specific versions to represent the core evolutionary steps:
1.  **Baseline Establishment (VGG16)**: Determining the upper bounds of accuracy regardless of size.
2.  **Efficiency Balancing (EfficientNetB1)**: Transitioning towards mobile-friendly architectures.
3.  **Edge Optimization (MobileNetV3)**: Finalizing the lightweight backbone with the best parameter-to-latency ratio.

The provided notebooks showcase the full journey of shaping the final model, from raw data processing to advanced hyperparameter tuning and quantization.

### 🧩 Reproducibility
To ensure consistent results across different environments, all experiments utilize **deterministic seeding**. We explicitly set seeds for `numpy`, `python.random`, and `tensorflow` to ensure that the training process and model weights remain reproducible.

### 📉 Compression & Quantization Nuances
While three different compression formats (Dynamic Range, Float16, and Int8) were generated, this project specifically utilizes the **Float16 (.tflite)** model for deployment. 
*   **Rationale**: Float16 provided the most stable performance. Other formats encountered compatibility issues with the **XNNPACK delegate**, and full Int8/Dynamic Range testing remains an area for future validation.

## 📊 Quick Links
*   **Dataset:** [Driver Inattention Detection Dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
*   **Live Notebook:** [CV-MiniProject on Kaggle](https://www.kaggle.com/code/saadaswad/cv-miniproject)

## 📈 Key Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Model Size (float16)** | **~2.00 MB** |
| **Inference Latency** | **~0.6 ms** (on CPU) |
| **Accuracy** | **~91%** |
| **Classes** | Aware, Drinking, Reckless, Sleepy, Yawning, Texting |

## 🛠 Tech Stack
- **Architecture:** MobileNetV3-Small (Fine-tuned)
- **Framework:** TensorFlow / Keras
- **Optimization:** TFLite (Post-Training Float16 Quantization)
- **Input Resolution:** 96x96 pixels


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
