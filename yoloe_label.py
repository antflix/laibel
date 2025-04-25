from ultralytics import YOLOE
import supervision as sv
import torch
from PIL import Image
import os

# --- Configuration ---
YOLOE_MODEL_PATH = "yoloe-11s-seg.pt" # Path relative to the script/app.py
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ["box"] # Class names the model can detect

yoloe_model = None
yoloe_model_load_error = None

def load_yoloe_model():
    """Loads the YOLOE model into the global 'yoloe_model' variable."""
    global yoloe_model, yoloe_model_load_error
    if yoloe_model:
        return True, None # Already loaded

    print(f"Attempting to load YOLOE model '{YOLOE_MODEL_PATH}' onto {DEVICE}...")
    try:
        if not os.path.exists(YOLOE_MODEL_PATH):
            error = f"YOLOE model file not found at {YOLOE_MODEL_PATH}"
            yoloe_model_load_error = error
            print(f"Error: {error}")
            return False, error

        model = YOLOE(YOLOE_MODEL_PATH).to(DEVICE)
        model.set_classes(NAMES, model.get_text_pe(NAMES)) # Important step for YOLOE
        yoloe_model = model # Assign to global only on success
        yoloe_model_load_error = None
        print(f"YOLOE model '{YOLOE_MODEL_PATH}' loaded successfully.")
        return True, None
    except Exception as e:
        error = f"Failed to load YOLOE model: {e}"
        yoloe_model_load_error = error
        yoloe_model = None
        print(f"Error: {error}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False, error

def predict_yoloe(image: Image.Image):
    """Runs prediction on a PIL image using the loaded YOLOE model."""
    if not yoloe_model:
        # Maybe attempt to load it here? Or rely on explicit loading.
        # Let's raise an error for clarity, requiring explicit load.
        raise RuntimeError("YOLOE model is not loaded. Call load_yoloe_model() first.")

    # Note: YOLOE might have specific input requirements, adjust if needed
    # The `.predict()` method usually handles necessary preprocessing.
    results = yoloe_model.predict(image, device=DEVICE, verbose=False) # verbose=False for cleaner logs

    if not results or not results[0]:
        return [] # No results or empty results array

    # Process results using supervision as before
    try:
        detections = sv.Detections.from_ultralytics(results[0])
    except Exception as e:
        print(f"Error processing YOLOE results with supervision: {e}")
        return [] # Return empty if processing fails

    # Return bounding box coordinates [[x1, y1, x2, y2], ...]
    # Ensure it handles cases where detections might exist but xyxy is empty
    return detections.xyxy.tolist() if detections.xyxy is not None and len(detections.xyxy) > 0 else []

# --- Example Usage (Optional, for testing the script directly) ---
# if __name__ == "__main__":
#     print(f"Running YOLOE Label script directly for testing...")
#     print(f"Using device: {DEVICE}")

#     # Attempt to load the model
#     loaded, error = load_yoloe_model()

#     if not loaded:
#         print(f"Model loading failed: {error}")
#     else:
#         print("Model loaded successfully for testing.")
#         # Create a dummy image or load a test image
#         try:
#             # Try loading a test image if it exists
#             test_image_path = "image.jpg" # Or provide a known test image path
#             if os.path.exists(test_image_path):
#                 print(f"Loading test image: {test_image_path}")
#                 test_image = Image.open(test_image_path).convert('RGB')

#                 # Perform prediction
#                 print("Running prediction on test image...")
#                 bounding_boxes = predict_yoloe(test_image)

#                 print("\n--- Detected Bounding Boxes (from test run) ---")
#                 if bounding_boxes:
#                     print(f"  Coordinates (x1, y1, x2, y2):")
#                     for box in bounding_boxes:
#                         print(f"    {box}")
#                 else:
#                     print("  No boxes detected!")
#             else:
#                 print(f"Test image '{test_image_path}' not found. Skipping prediction test.")

#         except FileNotFoundError:
#             print(f"Test image '{test_image_path}' not found. Skipping prediction test.")
#         except RuntimeError as e:
#             print(f"Runtime error during prediction test: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred during prediction test: {e}")
#             import traceback
#             traceback.print_exc()
