"""
Driver Behavior Classification using TFLite Model
Uses device camera to capture frames and classify driver behavior.
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# ============================================================
# CONFIGURATION
# ============================================================
CLASS_NAMES = [
    "DangerousDriving",  # 0
    "Distracted",        # 1
    "Drinking",          # 2
    "SafeDriving",       # 3
    "SleepyDriving",     # 4
    "Yawn",              # 5
]


MODEL_PATH = "demo\models\model_float16_128.tflite"  
# MODEL_PATH = "models/model.keras"  # Keras version (may have version compatibility issues)
INPUT_SIZE = (128, 128)  
CONFIDENCE_FALLBACK = 0.88  # If below, treat as distracted/concerned
USE_GRAYSCALE = True  
MODE = "interval"  # "continuous" for realtime; "interval" for periodic checks
INFERENCE_INTERVAL_SEC = 0.8  # Used when MODE == "interval"
CROP_MODE = "face"  # "face" to crop to largest detected face; "none" to use full frame
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_MARGIN = 0.35  # Expand face box to include head context

# ============================================================
# Custom Layers (for Keras version compatibility)
# ============================================================
@tf.keras.utils.register_keras_serializable()
class RandomContrastCompat(tf.keras.layers.RandomContrast):
    """RandomContrast layer with backward compatibility for value_range param."""
    def __init__(self, factor, value_range=None, seed=None, **kwargs):
        super().__init__(factor=factor, seed=seed, **kwargs)

@tf.keras.utils.register_keras_serializable()
class RandomBrightnessCompat(tf.keras.layers.RandomBrightness):
    """RandomBrightness layer with backward compatibility for value_range param."""
    def __init__(self, factor, value_range=None, seed=None, **kwargs):
        # Ignore value_range as it's not supported in some Keras versions
        super().__init__(factor=factor, seed=seed, **kwargs)



# ============================================================
# Load Model
# ============================================================
def load_keras_model(model_path):
    """Load Keras model with custom object handling for version compatibility."""
    custom_objects = {
        'RandomContrast': RandomContrastCompat,
        'RandomBrightness': RandomBrightnessCompat,
    }
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Trying with safe_mode=False...")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
    
    print("Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    return model


def load_tflite_model(model_path):
    """Load and initialize TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model loaded successfully!")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details


# ============================================================
# Preprocessing and Inference
# ============================================================
def preprocess_frame(frame, input_size, input_dtype=np.float32):
    """
    Preprocess frame for model input.
    - Resize to model input size
    - Convert BGR to RGB (and optional grayscale)
    - Convert to model's input dtype with pixel values in [0, 255] range
    
    Note: MobileNetV3 from Keras has a built-in rescale layer.
    The preprocess_input function is a pass-through, so we just need
    to provide float pixels in [0, 255] range.
    """

    frame_resized = cv2.resize(frame, input_size)
    
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    if USE_GRAYSCALE:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        frame_rgb = np.stack([gray, gray, gray], axis=-1)
    
    frame_input = np.expand_dims(frame_rgb, axis=0).astype(input_dtype)
    
    return frame_input


def detect_and_crop_face(frame):
    """Detect the largest face and return a cropped region; fallback to full frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    if len(faces) == 0:
        return frame  # Fallback if no face detected

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    margin_w = int(w * FACE_MARGIN)
    margin_h = int(h * FACE_MARGIN)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(frame.shape[1], x + w + margin_w)
    y2 = min(frame.shape[0], y + h + margin_h)

    return frame[y1:y2, x1:x2]

def run_inference_keras(model, input_data):
    """Run inference using Keras model."""
    output = model.predict(input_data, verbose=0)
    return output


def run_inference_tflite(interpreter, input_details, output_details, input_data):
    """Run inference using TFLite interpreter."""
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def get_prediction(output, class_names):
    """Get predicted class and confidence from model output."""
   
   
    probabilities = output[0]
    
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return predicted_class, class_names[predicted_class], confidence

# ============================================================
# Main Application
# ============================================================
def get_status_color(class_name):
    """Return color based on classification (BGR format)."""
    if class_name == "SafeDriving":
        return (0, 255, 0)  # Green for safe
    if class_name in ["Distracted", "SleepyDriving", "Yawn", "Drinking"]:
        return (0, 165, 255)  # Orange for distracted/concerned
    return (0, 0, 255)  # Red for dangerous/reckless


def main():
    # Load model
    use_tflite = MODEL_PATH.endswith('.tflite')
    
    if use_tflite:
        print("Loading TFLite model...")
        interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
        model = None
        input_dtype = input_details[0]['dtype']
    else:
        print("Loading Keras model...")
        model = load_keras_model(MODEL_PATH)
        interpreter, input_details, output_details = None, None, None
        input_dtype = np.float32

    # Load face detector
    global face_detector
    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_detector.empty():
        print("Warning: Could not load face cascade; falling back to full-frame classification.")
        global CROP_MODE
        CROP_MODE = "none"
    
    # Start video capture
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nPress 'q' to quit")
    print("-" * 40)
    
    last_label = "SafeDriving"
    last_confidence = 1.0
    last_color = get_status_color(last_label)
    last_inference_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        now = time.time()
        should_infer = (MODE == "continuous") or (now - last_inference_ts >= INFERENCE_INTERVAL_SEC)

        if should_infer:
            if CROP_MODE == "face":
                roi = detect_and_crop_face(frame)
            else:
                roi = frame

            input_data = preprocess_frame(roi, INPUT_SIZE, input_dtype)
            
            if use_tflite:
                output = run_inference_tflite(interpreter, input_details, output_details, input_data)
            else:
                output = run_inference_keras(model, input_data)
            
            class_idx, class_name, confidence = get_prediction(output, CLASS_NAMES)
            
            low_confidence = confidence < CONFIDENCE_FALLBACK
            if low_confidence:
                class_name = "Distracted"
                class_idx = CLASS_NAMES.index(class_name)
            
            color = get_status_color(class_name)

            last_label = class_name
            last_confidence = confidence
            last_color = color
            last_inference_ts = now
        else:
            class_name = last_label
            confidence = last_confidence
            color = last_color
        
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        

        h, w = frame.shape[:2]
        x = w - label_size[0] - 20
        y = 40
        
        cv2.rectangle(frame, (x - 10, y - label_size[1] - 10), (w - 10, y + 10), color, -1)
        
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow('Driver Behavior Classification', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed.")

if __name__ == "__main__":
    main()
