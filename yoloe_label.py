# venv/laibel/yoloe_label.py
from ultralytics import YOLOE
import supervision as sv
#import habana_frameworks.torch.core as htcore <-- Include if using Intel Gaudi
import torch
from PIL import Image
import os
import numpy as np
from typing import List

# --- Configuration ---
# REMOVED: YOLOE_MODEL_PATH = "yoloe-11s-seg.pt" # Path relative to the script/app.py
# We'll make the path configurable or keep it standard
YOLOE_DEFAULT_MODEL_PATH = "yoloe-11s-seg.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'hpu' if torch.hpu.is_available()  else 'cpu') <-- Include if using Intel Gaudi

yoloe_model = None
yoloe_model_load_error = None
yoloe_model_classes = [] # Store the class names the model was loaded with

# Modified function to accept label names and model path
def load_yoloe_model(label_names: List[str], model_path: str = YOLOE_DEFAULT_MODEL_PATH): # <-- Change list[str] to List[str]
    """Loads the YOLOE model into the global 'yoloe_model' variable
       and configures it with the provided label names."""
    global yoloe_model, yoloe_model_load_error, yoloe_model_classes
    if yoloe_model:
        # Check if the requested labels are the same as the loaded ones
        if set(label_names) == set(yoloe_model_classes):
            print(f"YOLOE model already loaded with the correct classes: {yoloe_model_classes}")
            return True, None # Already loaded with correct classes
        else:
            print(f"YOLOE model loaded with different classes ({yoloe_model_classes}). Reloading with {label_names}.")
            # Reset model to force reload with new classes
            yoloe_model = None
            yoloe_model_classes = []
            yoloe_model_load_error = None

    if not label_names:
        error = "Cannot load YOLOE model: No labels provided."
        print(f"Error: {error}")
        yoloe_model_load_error = error
        return False, error

    print(f"Attempting to load YOLOE model '{model_path}' onto {DEVICE} with classes: {label_names}...")
    try:
        if not os.path.exists(model_path):
            error = f"YOLOE model file not found at {model_path}"
            yoloe_model_load_error = error
            print(f"Error: {error}")
            return False, error

        model = YOLOE(model_path).to(DEVICE)
        # Use the provided label_names
        print(f"Setting YOLOE model classes: {label_names}")
        model.set_classes(label_names, model.get_text_pe(label_names)) # Important step for YOLOE

        yoloe_model = model # Assign to global only on success
        yoloe_model_classes = list(label_names) # Store the classes used
        yoloe_model_load_error = None
        print(f"YOLOE model '{model_path}' loaded successfully with classes: {yoloe_model_classes}.")
        return True, None
    except Exception as e:
        error = f"Failed to load YOLOE model: {e}"
        yoloe_model_load_error = error
        yoloe_model = None
        yoloe_model_classes = []
        print(f"Error: {error}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False, error

# Modified function to return coordinates and class IDs
def predict_yoloe(image: Image.Image):
    """Runs prediction on a PIL image using the loaded YOLOE model.
       Returns a list of dictionaries: [{'coords': [x1, y1, x2, y2], 'class_id': int}, ...]"""
    if not yoloe_model:
        raise RuntimeError("YOLOE model is not loaded. Call load_yoloe_model() first.")
    if not yoloe_model_classes:
         raise RuntimeError("YOLOE model is loaded but class names are missing.")


    print(f"Predicting with YOLOE model configured for classes: {yoloe_model_classes}")
    results = yoloe_model.predict(image, device=DEVICE, verbose=False)

    if not results or not results[0] or results[0].boxes is None: # Check if boxes exist
        print("YOLOE prediction returned no boxes.")
        return [] # No results or empty results array

    # Process results using supervision
    try:
        detections = sv.Detections.from_ultralytics(results[0])
        print(f"Supervision detected {len(detections)} items.")
        # Debug: print contents of detections
        # print("Detections object:", detections)
        # print("Detections xyxy:", detections.xyxy)
        # print("Detections class_id:", detections.class_id)
        # print("Detections confidence:", detections.confidence)

    except Exception as e:
        print(f"Error processing YOLOE results with supervision: {e}")
        import traceback
        traceback.print_exc()
        return [] # Return empty if processing fails

    predictions = []
    # Ensure class_id and xyxy are available and have the same length
    if detections.xyxy is not None and detections.class_id is not None and len(detections.xyxy) == len(detections.class_id):
        for i in range(len(detections.xyxy)):
            coords = detections.xyxy[i].astype(int).tolist() # Convert to int list [x1, y1, x2, y2]
            class_id = int(detections.class_id[i])         # Convert class_id to int
            if class_id < 0 or class_id >= len(yoloe_model_classes):
                 print(f"Warning: Skipping prediction with invalid class_id {class_id} (out of range for loaded classes {yoloe_model_classes})")
                 continue
            predictions.append({
                "coords": coords,
                "class_id": class_id
                # Optional: Add confidence if needed and available
                # "confidence": float(detections.confidence[i]) if detections.confidence is not None else None
            })
    else:
        print("Warning: Mismatch between xyxy and class_id counts or data missing in detections.")
        # Fallback or alternative processing if needed

    print(f"YOLOE predict_yoloe returning {len(predictions)} processed predictions.")
    return predictions

# --- Example Usage (Optional, for testing the script directly) ---
# (Keep the example usage commented out or update it if needed for direct testing)
# if __name__ == "__main__":
#     ...
