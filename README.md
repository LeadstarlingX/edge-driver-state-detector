# Driver Inattention Classifier

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
[![Notebook](https://img.shields.io/badge/Kaggle-Notebook-orange.svg)](https://www.kaggle.com/code/saadaswad/cv-miniproject)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time driver monitoring system using MobileNetV3. Edge-optimized for 0.6ms latency and 1.96MB size, achieving >90% accuracy in classifying inattentive states.

## 🚀 Overview
This repository contains a high-performance, lightweight Driver Monitoring System (DMS) designed for real-time edge deployment. The system classifies six distinct driver states to enhance road safety through proactive inattention detection.

## 📊 Quick Links
*   **Dataset:** [Driver Inattention Detection Dataset](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
*   **Live Notebook:** [CV-MiniProject on Kaggle](https://www.kaggle.com/code/saadaswad/cv-miniproject)
    > **Note:** You can view the full development history in the Kaggle notebook revisions, showing the progression from the initial baseline (VGG16) to the final optimized MobileNetV3 implementation.

## 📈 Key Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Model Size (FP16)** | **1.96 MB** |
| **Inference Latency** | **0.6 ms** (on CPU) |
| **Accuracy** | **~91%** |
| **Classes** | Aware, Drinking, Reckless, Sleepy, Yawning, Texting |

## 🛠 Tech Stack
- **Architecture:** MobileNetV3-Small (Fine-tuned)
- **Framework:** TensorFlow / Keras
- **Optimization:** TFLite (Post-Training Float16 Quantization)
- **Input Resolution:** 96x96 pixels

## 📄 Documentation
For a deep dive into the architecture evolution and optimization strategies, see the [Technical Report](driver_monitoring_system_report.md).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
