from flask import Flask, render_template, request, jsonify
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO # Import YOLO
import torch # Often needed implicitly by ultralytics

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here' # Keep this secure in production
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Still used for potential future uploads, though not directly now

# Ensure upload directory exists (optional if not using file system saves)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- AI Model State (Global) ---
MODEL_PATH = 'models/dome.pt' # Path relative to app.py
yolo_model = None
model_load_error = None
is_model_loading = False # Flag to prevent concurrent loading attempts
# --- End AI Model State ---

# --- Function to Load Model ---
def load_yolo_model():
    global yolo_model, model_load_error, is_model_loading
    if yolo_model:
        print("Model already loaded.")
        return True, None # Already loaded, success

    if is_model_loading:
        print("Model loading already in progress.")
        return False, "Model loading already in progress." # Loading in progress

    is_model_loading = True
    model_load_error = None # Reset error before attempting load
    print(f"Attempting to load YOLO model '{MODEL_PATH}'...")

    try:
        if not os.path.exists(MODEL_PATH):
            error = f"Model file not found at {MODEL_PATH}"
            print(f"Error: {error}")
            model_load_error = error
            is_model_loading = False
            return False, error

        # Check for CUDA device, fallback to CPU if not available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        temp_model = YOLO(MODEL_PATH)
        temp_model.to(device) # Move model to appropriate device
        yolo_model = temp_model # Assign to global variable only after successful loading
        print(f"YOLO model '{MODEL_PATH}' loaded successfully on {device}.")

        # Optional: Warm-up (consider if needed, adds time to load request)
        # try:
        #     dummy_img = Image.new('RGB', (640, 480), color = 'red')
        #     yolo_model.predict(dummy_img, verbose=False)
        #     print("Model warm-up successful.")
        # except Exception as warmup_err:
        #     print(f"Warning: Model warm-up failed: {warmup_err}")

        is_model_loading = False
        return True, None # Success

    except Exception as e:
        error = f"Failed to load YOLO model: {e}"
        print(f"Error: {error}")
        model_load_error = error
        yolo_model = None # Ensure model is None if loading failed
        is_model_loading = False
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False, error # Failure
    finally:
        is_model_loading = False # Ensure flag is reset even on unexpected exit
# --- End Function to Load Model ---

@app.route('/')
def index():
    # Pass model status to template
    # Check yolo_model directly, model_load_error might have transient errors from previous attempts
    return render_template('index.html', model_loaded=yolo_model is not None, model_error=model_load_error)

# --- New Endpoint to Trigger Model Loading ---
@app.route('/load_model', methods=['POST'])
def trigger_load_model():
    if yolo_model:
        return jsonify({"success": True, "message": "Model already loaded."})
    if is_model_loading:
        return jsonify({"success": False, "error": "Model loading already in progress."}), 429 # Too Many Requests

    success, error_message = load_yolo_model()

    if success:
        return jsonify({"success": True, "message": "Model loaded successfully."})
    else:
        # Include the specific error message
        return jsonify({"success": False, "error": error_message or "Failed to load model."}), 500
# --- End New Endpoint ---

# This route is kept for potential future use but not strictly needed for current flow
@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print("Received data for saving (placeholder):", data)
    # Here you would save the annotation data persistently if needed
    return jsonify({"success": True, "message": "Annotation save endpoint reached (placeholder)"})

# --- AI Assist Endpoint ---
@app.route('/ai_assist', methods=['POST'])
def ai_assist():
    # CRITICAL: Check if model is loaded before attempting inference
    if not yolo_model:
        error_msg = "AI model is not loaded."
        if model_load_error:
             error_msg += f" Last known error: {model_load_error}"
        return jsonify({"success": False, "error": error_msg }), 503 # Service Unavailable

    try:
        data = request.json
        if 'image_data' not in data:
            return jsonify({"success": False, "error": "Missing image_data in request"}), 400

        image_data_url = data['image_data']
        # Decode Base64 image data URL (e.g., "data:image/png;base64,iVBOR...")
        try:
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes)).convert('RGB') # Ensure RGB
        except Exception as decode_err:
            print(f"Error decoding image: {decode_err}")
            return jsonify({"success": False, "error": f"Invalid image data format: {decode_err}"}), 400

        print(f"Performing AI inference on image of size {image.size}...")

        # Perform prediction
        results = yolo_model.predict(image, conf=0.25, verbose=False)

        detected_boxes = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            class_names = result.names

            if boxes is not None:
                print(f"Detected {len(boxes)} potential objects.")
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_index = int(box.cls[0].cpu().numpy())
                    label = class_names.get(cls_index, f"class_{cls_index}")

                    detected_boxes.append({
                        "x_min": int(xyxy[0]),
                        "y_min": int(xyxy[1]),
                        "x_max": int(xyxy[2]),
                        "y_max": int(xyxy[3]),
                        "label": label,
                        "confidence": round(conf, 3)
                    })
            else:
                print("No boxes found in the results.")
        else:
             print("Inference returned no results or unexpected format.")


        print(f"AI Assist finished. Found {len(detected_boxes)} boxes above threshold.")
        return jsonify({
            "success": True,
            "boxes": detected_boxes
        })

    except Exception as e:
        print(f"Error during AI Assist processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to Flask console
        return jsonify({"success": False, "error": f"Internal server error during inference: {e}"}), 500
# --- End AI Assist Endpoint ---

if __name__ == '__main__':
    # use_reloader=False is important now to prevent the model state (yolo_model)
    # from being reset unexpectedly during development with debug=True.
    app.run(debug=True, use_reloader=False)
