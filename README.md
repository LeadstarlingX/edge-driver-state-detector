# Driver Inattention Classifier

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset)
[![Notebook](https://img.shields.io/badge/Kaggle-Notebook-orange.svg)](https://www.kaggle.com/code/saadaswad/cv-miniproject)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time driver monitoring system using MobileNetV3. Edge-optimized for 0.6ms latency and 1.96MB size, achieving >90% accuracy in classifying inattentive states.

## 🚀 Methodology & Evolution
The development of this system followed an iterative approach, documented through multiple notebook versions. We have selected specific versions to represent the core evolutionary steps:
1.  **Baseline Establishment (VGG16)**: Determining the upper bounds of accuracy regardless of size.
2.  **Efficiency Balancing (EfficientNetB1)**: Transitioning towards mobile-friendly architectures.
3.  **Edge Optimization (MobileNetV3)**: Finalizing the lightweight backbone. We iterated through three distinct refinements (visible in `v26` notebook) to maximize performance:
    *   **Base**: Standard transfer learning (`model_float16`).
    *   **Tuned**: Fine-tuned with selective unfreezing (`model_float16_tuned`).
    *   **Enhanced (Final)**: The deployed model (`model_float16_128`) incorporating:
        *   **Resolution Increase**: 96x96 → 128x128.
        *   **Advanced Training**: Per-class dataset balancing, label smoothing, and aggressive augmentation.
        *   **Result**: Accuracy improved to **94%** with high precision across all classes.

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
| **Accuracy** | **94%** |
| **Classes** | Aware, Drinking, Reckless, Sleepy, Yawning, Texting |

### 🔬 Evolution of the Final Model
To achieve the final 94% accuracy, we performed an ablation study on the MobileNetV3 architecture. The impact of each optimization step is detailed below:

| Configuration | Accuracy | Macro Avg F1 | Weighted Avg F1 |
| :--- | :--- | :--- | :--- |
| **1. Per-class Balancing** | 92% | 0.90 | 0.92 |
| **2. + Label Smoothing** | 92% | 0.90 | 0.92 |
| **3. + 128x128 Resolution** | **94%** | **0.92** | **0.94** |

<details>
<summary><strong>View Detailed Classification Reports</strong></summary>

#### 1. Only Per-class Dataset Balancing
```text
                 precision    recall  f1-score   support

DangerousDriving       0.99      0.99      0.99       301
      Distracted       0.87      0.86      0.86       152
        Drinking       0.92      0.96      0.94        25
     SafeDriving       0.93      0.90      0.92       412
   SleepyDriving       0.72      0.86      0.78        69
            Yawn       0.86      0.96      0.91        26

        accuracy                           0.92       985
       macro avg       0.88      0.92      0.90       985
    weighted avg       0.92      0.92      0.92       985
```

#### 2. Balancing + Label Smoothing
```text
                 precision    recall  f1-score   support

DangerousDriving       0.99      1.00      1.00       301
      Distracted       0.88      0.83      0.85       152
        Drinking       0.96      0.88      0.92        25
     SafeDriving       0.93      0.90      0.91       412
   SleepyDriving       0.66      0.88      0.76        69
            Yawn       0.93      0.96      0.94        26

        accuracy                           0.92       985
       macro avg       0.89      0.91      0.90       985
    weighted avg       0.92      0.92      0.92       985
```

#### 3. Balancing + Label Smoothing + 128x128 (Final)
```text
                 precision    recall  f1-score   support

DangerousDriving       0.99      0.99      0.99       301
      Distracted       0.87      0.84      0.85       152
        Drinking       1.00      0.88      0.94        25
     SafeDriving       0.94      0.93      0.94       412
   SleepyDriving       0.78      0.94      0.86        69
            Yawn       0.93      1.00      0.96        26

        accuracy                           0.94       985
       macro avg       0.92      0.93      0.92       985
    weighted avg       0.94      0.94      0.94       985
```
</details>

## � Real-time Demo
A Python-based demonstration is provided in the `demo/` folder to validate the model in a real-world setting. It uses the requested TFLite model to perform inference on a live webcam feed.

**Features:**
*   **Real-time Inference**: Uses the optimized `model_float16_128.tflite` by default.
*   **Preprocessing**: Automatically handles face detection, cropping, and resizing to 128x128.
*   **Visualization**: Displays the live feed with predicted class overlays and confidence scores.

To run the demo:
```bash
pip install opencv-python tensorflow  # or tflite-runtime
python demo/main.py
```

## �🛠 Tech Stack
- **Architecture:** MobileNetV3-Small (Fine-tuned)
- **Framework:** TensorFlow / Keras
- **Optimization:** TFLite (Post-Training Float16 Quantization)
- **Input Resolution:** 128x128 pixels

## 🤝 Feedback & Contributions

Please use the project's **[Kaggle Discussion](https://www.kaggle.com/code/saadaswad/cv-miniproject)** to share:
- Technical insights on architecture or quantization
- Edge deployment optimizations
- Suggestions for improving code or documentation


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
