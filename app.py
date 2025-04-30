# venv/laibel/app.py
from flask import Flask, render_template, request, jsonify
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# Import the YOLOE module itself, and specific functions/vars if needed elsewhere
import yoloe_label
# Use the modified load function name directly
from yoloe_label import load_yoloe_model as load_yoloe_model_with_labels, predict_yoloe

import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- AI Model State (Global) ---
YOLO_MODEL_PATH = 'models/dome.pt'
yolo_model = None
yolo_model_load_error = None
is_model_loading = False

# --- YOLOE Model State ---
is_yoloe_loading = False
# We now rely on yoloe_label.yoloe_model and yoloe_label.yoloe_model_classes
# --- End AI Model State ---

# --- Function to Load YOLO Model ---
# (Keep the existing load_yolo_model function as is)
def load_yolo_model():
    global yolo_model, yolo_model_load_error, is_model_loading
    if yolo_model:
        print("YOLO Model already loaded.")
        return True, None # Already loaded, success

    if is_model_loading:
        print("YOLO Model loading already in progress.")
        return False, "YOLO Model loading already in progress." # Loading in progress

    is_model_loading = True
    yolo_model_load_error = None # Reset error before attempting load
    print(f"Attempting to load YOLO model '{YOLO_MODEL_PATH}'...")

    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            error = f"YOLO Model file not found at {YOLO_MODEL_PATH}"
            print(f"Error: {error}")
            yolo_model_load_error = error
            is_model_loading = False # Reset flag on early exit
            return False, error

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device} for YOLO model")

        temp_model = YOLO(YOLO_MODEL_PATH)
        temp_model.to(device)
        yolo_model = temp_model
        print(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully on {device}.")
        yolo_model_load_error = None

        return True, None # Success

    except Exception as e:
        error = f"Failed to load YOLO model: {e}"
        print(f"Error: {error}")
        yolo_model_load_error = error
        yolo_model = None
        import traceback
        traceback.print_exc()
        return False, error
    finally:
        is_model_loading = False
# --- End Function to Load YOLO Model ---

@app.route('/')
def index():
    # Pass model status directly from the source module/variable
    # Also pass the loaded YOLOE classes if available
    yoloe_classes = getattr(yoloe_label, 'yoloe_model_classes', [])
    print(f"Rendering index. YOLOE Status - Model: {yoloe_label.yoloe_model is not None}, Error: {yoloe_label.yoloe_model_load_error}, Classes: {yoloe_classes}")
    return render_template(
        'index.html',
        yolo_model_loaded=yolo_model is not None,
        yolo_model_error=yolo_model_load_error,
        # Check the model directly within the yoloe_label module's namespace
        yoloe_model_loaded=yoloe_label.yoloe_model is not None,
        yoloe_model_error=yoloe_label.yoloe_model_load_error,
        yoloe_model_classes=yoloe_classes # Pass classes to template/JS if needed
    )

# --- Endpoint to Trigger YOLOE Model Loading (MODIFIED) ---
@app.route('/load_yoloe_model', methods=['POST'])
def trigger_load_yoloe_model():
    global is_yoloe_loading
    data = request.json
    requested_labels = data.get('labels') if data else None

    if not requested_labels:
         print("Load YOLOE request failed: No labels provided in request body.")
         return jsonify({"success": False, "error": "No labels provided for YOLOE model initialization."}), 400

    print(f"Received request to load YOLOE model with labels: {requested_labels}")

    # Check if model is loaded *and* has the same classes
    if yoloe_label.yoloe_model and set(requested_labels) == set(yoloe_label.yoloe_model_classes):
        print(f"YOLOE model already loaded with the requested classes: {requested_labels}")
        return jsonify({"success": True, "message": f"YOLOE model already loaded with classes: {', '.join(requested_labels)}"})

    if is_yoloe_loading:
        print("YOLOE model loading already in progress.")
        return jsonify({"success": False, "error": "YOLOE model loading already in progress."}), 429 # Too Many Requests

    is_yoloe_loading = True
    # Call the modified loading function, passing the labels
    # Assuming default model path for now, could make it configurable
    print(f"Calling load_yoloe_model_with_labels with: {requested_labels}")
    success, error_message = load_yoloe_model_with_labels(requested_labels)
    is_yoloe_loading = False

    if success:
        loaded_classes = getattr(yoloe_label, 'yoloe_model_classes', [])
        print(f"YOLOE model loaded successfully. Classes set to: {loaded_classes}")
        return jsonify({"success": True, "message": f"YOLOE model loaded successfully with classes: {', '.join(loaded_classes)}"})
    else:
        print(f"YOLOE model loading failed: {error_message}")
        return jsonify({"success": False, "error": error_message or "Failed to load YOLOE model."}), 500

# --- Endpoint to Trigger YOLO Model Loading ---
# (Keep the existing trigger_load_yolo_model function as is)
@app.route('/load_yolo_model', methods=['POST'])
def trigger_load_yolo_model():
    if yolo_model:
        return jsonify({"success": True, "message": "YOLO Model already loaded."})
    if is_model_loading:
        return jsonify({"success": False, "error": "YOLO Model loading already in progress."}), 429

    success, error_message = load_yolo_model()

    if success:
        return jsonify({"success": True, "message": "YOLO Model loaded successfully."})
    else:
        return jsonify({"success": False, "error": error_message or "Failed to load YOLO model."}), 500
# --- End YOLO Load Endpoint ---

# --- Save Annotation Placeholder ---
@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print("Received data for saving (placeholder):", data)
    return jsonify({"success": True, "message": "Annotation save endpoint reached (placeholder)"})

# --- AI Assist Endpoint (YOLO) ---
# (Keep the existing ai_assist function as is)
@app.route('/ai_assist', methods=['POST'])
def ai_assist():
    if not yolo_model:
        error_msg = "YOLO AI model is not loaded."
        if yolo_model_load_error:
             error_msg += f" Last known error: {yolo_model_load_error}"
        return jsonify({"success": False, "error": error_msg }), 503

    try:
        data = request.json
        if 'image_data' not in data:
            return jsonify({"success": False, "error": "Missing image_data in request"}), 400

        image_data_url = data['image_data']
        try:
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as decode_err:
            print(f"Error decoding image for YOLO: {decode_err}")
            return jsonify({"success": False, "error": f"Invalid image data format: {decode_err}"}), 400

        print(f"Performing YOLO AI inference on image of size {image.size}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = yolo_model.predict(image, conf=0.25, verbose=False, device=device)

        detected_boxes = []
        if results and len(results) > 0:
            result = results[0]
            # Ensure boxes and names are present
            boxes = getattr(result, 'boxes', None)
            class_names = getattr(result, 'names', {}) # Use getattr for safety

            if boxes is not None:
                print(f"YOLO Detected {len(boxes)} potential objects.")
                for box in boxes:
                    # Safely access attributes, converting tensors to CPU and numpy/float/int
                    try:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_index = int(box.cls[0].cpu().numpy())
                        label = class_names.get(cls_index, f"class_{cls_index}") # Get label from names dict

                        detected_boxes.append({
                            "x_min": int(xyxy[0]), "y_min": int(xyxy[1]),
                            "x_max": int(xyxy[2]), "y_max": int(xyxy[3]),
                            "label": label, "confidence": round(conf, 3)
                        })
                    except (AttributeError, IndexError, TypeError) as box_err:
                         print(f"Warning: Skipping YOLO box due to processing error: {box_err}. Box data: {box}")

            else:
                print("No boxes found in the YOLO results object.")
        else:
             print("YOLO Inference returned no results or unexpected format.")

        print(f"YOLO AI Assist finished. Found {len(detected_boxes)} boxes above threshold.")
        return jsonify({"success": True, "boxes": detected_boxes})

    except Exception as e:
        print(f"Error during YOLO AI Assist processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error during YOLO inference: {e}"}), 500
# --- End AI Assist Endpoint ---


# --- YOLOE Assist Endpoint (MODIFIED) ---
@app.route('/yoloe_assist', methods=['POST'])
def yoloe_assist():
    # CRITICAL: Check the model directly in the yoloe_label module's namespace
    print(f"Checking YOLOE model status in /yoloe_assist: Model loaded = {yoloe_label.yoloe_model is not None}")
    if not yoloe_label.yoloe_model:
        error_msg = "YOLOE AI model is not loaded."
        # Check the error variable directly in the module as well
        if yoloe_label.yoloe_model_load_error:
            error_msg += f" Last known error: {yoloe_label.yoloe_model_load_error}"
        print(f"YOLOE model check failed in /yoloe_assist. Error: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 503 # Service Unavailable

    # Also check if the classes were loaded correctly
    loaded_classes = getattr(yoloe_label, 'yoloe_model_classes', [])
    if not loaded_classes:
        error_msg = "YOLOE model is loaded, but its class list is missing or empty."
        print(f"YOLOE assist check failed: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500 # Internal Server Error or 503?

    print(f"YOLOE model ready for prediction with classes: {loaded_classes}")

    try:
        data = request.json
        if 'image_data' not in data:
            return jsonify({"success": False, "error": "Missing image_data in request"}), 400

        image_data_url = data['image_data']
        try:
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as decode_err:
            print(f"Error decoding image for YOLOE: {decode_err}")
            return jsonify({"success": False, "error": f"Invalid image data format: {decode_err}"}), 400

        print(f"Performing YOLOE inference on image of size {image.size}...")

        # Perform prediction using the imported function
        # This now returns [{'coords': [x1,y1,x2,y2], 'class_id': id}, ...]
        predictions = predict_yoloe(image)

        detected_boxes = []
        if predictions:
            print(f"YOLOE raw predictions received: {len(predictions)}")
            for pred in predictions:
                coords = pred.get('coords')
                class_id = pred.get('class_id')

                # Validate coordinates and class_id
                if not (coords and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
                    print(f"Warning: Skipping invalid coordinate set from YOLOE: {coords}")
                    continue
                if not isinstance(class_id, int) or class_id < 0 or class_id >= len(loaded_classes):
                    print(f"Warning: Skipping prediction with invalid class_id: {class_id} (available: {len(loaded_classes)})")
                    continue

                # Get the label name using the class_id and loaded_classes
                label_name = loaded_classes[class_id]

                detected_boxes.append({
                    "x_min": int(coords[0]), "y_min": int(coords[1]),
                    "x_max": int(coords[2]), "y_max": int(coords[3]),
                    "label": label_name, # Use the dynamically determined label
                    "confidence": None # Confidence not handled in this version yet
                })

            print(f"Processed {len(detected_boxes)} valid YOLOE bounding boxes.")
        else:
            print("YOLOE Inference returned no valid predictions.")

        print(f"YOLOE Assist finished. Found {len(detected_boxes)} boxes.")
        return jsonify({"success": True, "boxes": detected_boxes})

    # Catch the specific RuntimeError raised by predict_yoloe if model not loaded/configured
    except RuntimeError as e:
         print(f"RuntimeError during YOLOE Assist: {e}")
         # Make sure model state reflects the error if it indicates model not loaded
         if "model is not loaded" in str(e).lower():
             yoloe_label.yoloe_model = None
             yoloe_label.yoloe_model_classes = []
         return jsonify({"success": False, "error": str(e)}), 500 # Or 503 if model not loaded
    except Exception as e:
        print(f"Error during YOLOE Assist processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error during YOLOE inference: {e}"}), 500
# --- End YOLOE Assist Endpoint ---

if __name__ == '__main__':
    # use_reloader=False is important!
    app.run(debug=False, use_reloader=False)
