from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import base64
import yaml
import zipfile
import shutil
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import glob
from pathlib import Path

# Import the YOLOE module itself, and specific functions/vars if needed elsewhere
import yoloe_label
# Use the modified load function name directly
from yoloe_label import load_yoloe_model as load_yoloe_model_with_labels, predict_yoloe

import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
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

# --- Dataset Management Functions ---
def scan_datasets():
    """Scan the uploads directory for YOLO datasets"""
    datasets = []
    uploads_path = Path(app.config['UPLOAD_FOLDER'])
    
    print(f"Scanning for datasets in: {uploads_path}")
    
    if not uploads_path.exists():
        print("Uploads folder doesn't exist, creating it...")
        uploads_path.mkdir(parents=True, exist_ok=True)
        return datasets
    
    # Look for directories in uploads folder
    for item in uploads_path.iterdir():
        if item.is_dir():
            print(f"Found directory: {item.name}")
            dataset_info = analyze_dataset(item)
            if dataset_info:
                datasets.append(dataset_info)
                print(f"Valid dataset found: {item.name} with {dataset_info['total_images']} images")
            else:
                print(f"Invalid dataset: {item.name}")
    
    print(f"Total valid datasets found: {len(datasets)}")
    return datasets

def analyze_dataset(dataset_path):
    """Analyze a directory to determine if it's a valid YOLO dataset"""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    print(f"\
Analyzing dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    
    # Look for data.yaml or similar configuration file
    yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
    
    dataset_info = {
        'name': dataset_name,
        'path': str(dataset_path),
        'yaml_file': None,
        'classes': [],
        'splits': {},
        'total_images': 0,
        'total_labels': 0,
        'valid': False
    }
    
    # Parse YAML file if exists
    if yaml_files:
        yaml_file = yaml_files[0]  # Use the first YAML file found
        dataset_info['yaml_file'] = str(yaml_file)
        print(f"Found YAML file: {yaml_file}")
        
        try:
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            print(f"YAML contents: {yaml_data}")
            
            # Extract classes
            if 'names' in yaml_data:
                if isinstance(yaml_data['names'], dict):
                    dataset_info['classes'] = list(yaml_data['names'].values())
                elif isinstance(yaml_data['names'], list):
                    dataset_info['classes'] = yaml_data['names']
                print(f"Classes from YAML: {dataset_info['classes']}")
            
            # Extract split information
            for split in ['train', 'val', 'test', 'valid']:
                if split in yaml_data:
                    split_path = yaml_data[split]
                    if isinstance(split_path, str):
                        # Handle relative paths
                        if not os.path.isabs(split_path):
                            if split_path.endswith('/images') or split_path.endswith('\\\\images'):
                                # Path points to images directory
                                full_split_path = dataset_path / split_path
                                images_dir = full_split_path
                                labels_dir = full_split_path.parent / 'labels'
                            else:
                                # Path points to split directory, look for images/labels subdirs
                                full_split_path = dataset_path / split_path
                                images_dir = full_split_path / 'images' if (full_split_path / 'images').exists() else full_split_path
                                labels_dir = full_split_path / 'labels' if (full_split_path / 'labels').exists() else None
                        else:
                            split_path = Path(split_path)
                            images_dir = split_path / 'images' if (split_path / 'images').exists() else split_path
                            labels_dir = split_path / 'labels' if (split_path / 'labels').exists() else None
                        
                        print(f"Split '{split}' - Images dir: {images_dir}, Labels dir: {labels_dir}")
                        
                        if images_dir and images_dir.exists():
                            # Count images using proper file extensions
                            image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
                            images = []
                            
                            for ext in image_extensions:
                                # Use rglob for recursive search, but limit to immediate directory for performance
                                images.extend(list(images_dir.glob(f'*.{ext}')))
                                images.extend(list(images_dir.glob(f'*.{ext.upper()}')))
                            
                            labels = []
                            if labels_dir and labels_dir.exists():
                                labels = list(labels_dir.glob('*.txt'))
                            
                            print(f"Found {len(images)} images and {len(labels)} labels in {split}")
                            
                            if images:  # Only add if we found images
                                dataset_info['splits'][split] = {
                                    'images_dir': str(images_dir),
                                    'labels_dir': str(labels_dir) if labels_dir and labels_dir.exists() else None,
                                    'image_count': len(images),
                                    'label_count': len(labels),
                                    'images': [img.name for img in images]
                                }
                                
                                dataset_info['total_images'] += len(images)
                                dataset_info['total_labels'] += len(labels)
            
            dataset_info['valid'] = dataset_info['total_images'] > 0
            
        except Exception as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
    
    # If no YAML or YAML didn't work, try to infer structure
    if not dataset_info['valid']:
        print("No valid YAML found, trying to infer dataset structure...")
        
        # Look for common YOLO directory structures
        for subdir in ['train', 'val', 'test', 'valid']:
            subdir_path = dataset_path / subdir
            print(f"Checking for split directory: {subdir_path}")
            
            if subdir_path.exists() and subdir_path.is_dir():
                print(f"Found split directory: {subdir}")
                
                # Check for images subdirectory, otherwise use the split directory itself
                images_dir = subdir_path / 'images' if (subdir_path / 'images').exists() else subdir_path
                labels_dir = subdir_path / 'labels' if (subdir_path / 'labels').exists() else None
                
                print(f"Images dir: {images_dir}, Labels dir: {labels_dir}")
                
                # Count images
                image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
                images = []
                
                for ext in image_extensions:
                    images.extend(list(images_dir.glob(f'*.{ext}')))
                    images.extend(list(images_dir.glob(f'*.{ext.upper()}')))
                
                labels = []
                if labels_dir and labels_dir.exists():
                    labels = list(labels_dir.glob('*.txt'))
                
                print(f"Found {len(images)} images and {len(labels)} labels")
                
                if images:  # Only add if we found images
                    dataset_info['splits'][subdir] = {
                        'images_dir': str(images_dir),
                        'labels_dir': str(labels_dir) if labels_dir and labels_dir.exists() else None,
                        'image_count': len(images),
                        'label_count': len(labels),
                        'images': [img.name for img in images]
                    }
                    
                    dataset_info['total_images'] += len(images)
                    dataset_info['total_labels'] += len(labels)
        
        # Try to infer classes from label files
        if dataset_info['splits'] and not dataset_info['classes']:
            print("Inferring classes from label files...")
            classes_set = set()
            for split_info in dataset_info['splits'].values():
                labels_dir = split_info['labels_dir']
                if labels_dir and os.path.exists(labels_dir):
                    for label_file in Path(labels_dir).glob('*.txt'):
                        try:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        class_id = int(parts[0])
                                        classes_set.add(class_id)
                        except Exception as e:
                            print(f"Error reading label file {label_file}: {e}")
                            continue
            
            if classes_set:
                max_class = max(classes_set)
                dataset_info['classes'] = [f'class_{i}' for i in range(max_class + 1)]
                print(f"Inferred classes: {dataset_info['classes']}")
        
        dataset_info['valid'] = dataset_info['total_images'] > 0
    
    print(f"Dataset analysis complete. Valid: {dataset_info['valid']}, Total images: {dataset_info['total_images']}")
    return dataset_info if dataset_info['valid'] else None

def load_dataset_images(dataset_name, split=None):
    """Load images from a specific dataset and split"""
    datasets = scan_datasets()
    dataset = next((d for d in datasets if d['name'] == dataset_name), None)
    
    if not dataset:
        return []
    
    images = []
    splits_to_load = [split] if split else dataset['splits'].keys()
    
    for split_name in splits_to_load:
        if split_name in dataset['splits']:
            split_info = dataset['splits'][split_name]
            images_dir = split_info['images_dir']
            labels_dir = split_info['labels_dir']
            
            for image_name in split_info['images']:
                image_path = os.path.join(images_dir, image_name)
                label_path = None
                
                if labels_dir:
                    label_name = os.path.splitext(image_name)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_name)
                    if not os.path.exists(label_path):
                        label_path = None
                
                if os.path.exists(image_path):
                    images.append({
                        'name': image_name,
                        'image_path': image_path,
                        'label_path': label_path,
                        'split': split_name,
                        'dataset': dataset_name
                    })
    
    return images

def parse_yolo_label(label_path, image_width, image_height, class_names):
    """Parse a YOLO format label file"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert from YOLO format (normalized) to pixel coordinates
                    x = (x_center - width / 2) * image_width
                    y = (y_center - height / 2) * image_height
                    w = width * image_width
                    h = height * image_height
                    
                    class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                    
                    boxes.append({
                        'x': max(0, x),
                        'y': max(0, y),
                        'width': min(w, image_width - x),
                        'height': min(h, image_height - y),
                        'label': class_name
                    })
    except Exception as e:
        print(f"Error parsing label file {label_path}: {e}")
    
    return boxes

# --- Function to Load YOLO Model ---
def load_yolo_model():
    global yolo_model, yolo_model_load_error, is_model_loading
    if yolo_model:
        print("YOLO Model already loaded.")
        return True, None

    if is_model_loading:
        print("YOLO Model loading already in progress.")
        return False, "YOLO Model loading already in progress."

    is_model_loading = True
    yolo_model_load_error = None
    print(f"Attempting to load YOLO model '{YOLO_MODEL_PATH}'...")

    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            error = f"YOLO Model file not found at {YOLO_MODEL_PATH}"
            print(f"Error: {error}")
            yolo_model_load_error = error
            is_model_loading = False
            return False, error

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device} for YOLO model")

        temp_model = YOLO(YOLO_MODEL_PATH)
        temp_model.to(device)
        yolo_model = temp_model
        print(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully on {device}.")
        yolo_model_load_error = None

        return True, None

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

@app.route('/')
def index():
    yoloe_classes = getattr(yoloe_label, 'yoloe_model_classes', [])
    datasets = scan_datasets()
    
    print(f"Rendering index. YOLOE Status - Model: {yoloe_label.yoloe_model is not None}, Error: {yoloe_label.yoloe_model_load_error}, Classes: {yoloe_classes}")
    print(f"Found {len(datasets)} datasets: {[d['name'] for d in datasets]}")
    
    return render_template(
        'index.html',
        yolo_model_loaded=yolo_model is not None,
        yolo_model_error=yolo_model_load_error,
        yoloe_model_loaded=yoloe_label.yoloe_model is not None,
        yoloe_model_error=yoloe_label.yoloe_model_load_error,
        yoloe_model_classes=yoloe_classes,
        datasets=datasets
    )

# --- Dataset Management Endpoints ---
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    datasets = scan_datasets()
    return jsonify({"success": True, "datasets": datasets})

@app.route('/api/datasets/<dataset_name>/images', methods=['GET'])
def get_dataset_images(dataset_name):
    """Get images from a specific dataset"""
    split = request.args.get('split', None)
    images = load_dataset_images(dataset_name, split)
    return jsonify({"success": True, "images": images})

@app.route('/api/datasets/<dataset_name>/image/<path:image_path>')
def get_dataset_image(dataset_name, image_path):
    """Serve an image from a dataset"""
    try:
        # Security check: ensure the path is within our uploads directory
        full_path = os.path.abspath(image_path)
        uploads_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
        
        if not full_path.startswith(uploads_path):
            return jsonify({"error": "Invalid path"}), 403
        
        if os.path.exists(full_path):
            return send_file(full_path)
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/datasets/<dataset_name>/labels/<path:image_name>')
def get_image_labels(dataset_name, image_name):
    """Get YOLO labels for a specific image"""
    try:
        datasets = scan_datasets()
        dataset = next((d for d in datasets if d['name'] == dataset_name), None)
        
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        # Find the image in the dataset
        for split_name, split_info in dataset['splits'].items():
            if image_name in split_info['images']:
                if split_info['labels_dir']:
                    label_name = os.path.splitext(image_name)[0] + '.txt'
                    label_path = os.path.join(split_info['labels_dir'], label_name)
                    
                    # Load image to get dimensions
                    image_path = os.path.join(split_info['images_dir'], image_name)
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            image_width, image_height = img.size
                        
                        boxes = parse_yolo_label(label_path, image_width, image_height, dataset['classes'])
                        return jsonify({
                            "success": True,
                            "boxes": boxes,
                            "classes": dataset['classes']
                        })
        
        return jsonify({"success": True, "boxes": [], "classes": dataset['classes']})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload (ZIP files)"""
    try:
        print("Dataset upload request received")
        
        if 'dataset' not in request.files:
            print("No dataset file in request")
            return jsonify({"error": "No dataset file provided"}), 400
        
        file = request.files['dataset']
        if file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.zip'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Only ZIP files are supported"}), 400
        
        filename = secure_filename(file.filename)
        dataset_name = os.path.splitext(filename)[0]
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
        
        print(f"Uploading dataset: {filename} to {dataset_path}")
        
        # Remove existing dataset if it exists
        if os.path.exists(dataset_path):
            print(f"Removing existing dataset at {dataset_path}")
            shutil.rmtree(dataset_path)
        
        os.makedirs(dataset_path, exist_ok=True)
        
        # Extract ZIP file
        print("Extracting ZIP file...")
        try:
            with zipfile.ZipFile(file.stream, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print("ZIP extraction completed")
        except Exception as e:
            print(f"Error extracting ZIP: {e}")
            return jsonify({"error": f"Failed to extract ZIP file: {str(e)}"}), 400
        
        # Analyze the uploaded dataset
        print("Analyzing uploaded dataset...")
        dataset_info = analyze_dataset(dataset_path)
        
        if dataset_info:
            print(f"Dataset upload successful: {dataset_info}")
            return jsonify({
                "success": True,
                "message": f"Dataset '{dataset_name}' uploaded successfully",
                "dataset": dataset_info
            })
        else:
            print("Dataset analysis failed - removing uploaded files")
            shutil.rmtree(dataset_path)  # Clean up invalid dataset
            return jsonify({
                "error": "Invalid dataset structure. Please ensure it follows YOLO format with proper directory structure and image files."
            }), 400
    
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/load_yoloe_model', methods=['POST'])
def trigger_load_yoloe_model():
    global is_yoloe_loading
    data = request.json
    requested_labels = data.get('labels') if data else None

    if not requested_labels:
         print("Load YOLOE request failed: No labels provided in request body.")
         return jsonify({"success": False, "error": "No labels provided for YOLOE model initialization."}), 400

    print(f"Received request to load YOLOE model with labels: {requested_labels}")

    if yoloe_label.yoloe_model and set(requested_labels) == set(yoloe_label.yoloe_model_classes):
        print(f"YOLOE model already loaded with the requested classes: {requested_labels}")
        return jsonify({"success": True, "message": f"YOLOE model already loaded with classes: {', '.join(requested_labels)}"})

    if is_yoloe_loading:
        print("YOLOE model loading already in progress.")
        return jsonify({"success": False, "error": "YOLOE model loading already in progress."}), 429

    is_yoloe_loading = True
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

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print("Received data for saving (placeholder):", data)
    return jsonify({"success": True, "message": "Annotation save endpoint reached (placeholder)"})

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
            boxes = getattr(result, 'boxes', None)
            class_names = getattr(result, 'names', {})

            if boxes is not None:
                print(f"YOLO Detected {len(boxes)} potential objects.")
                for box in boxes:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_index = int(box.cls[0].cpu().numpy())
                        label = class_names.get(cls_index, f"class_{cls_index}")

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

@app.route('/yoloe_assist', methods=['POST'])
def yoloe_assist():
    print(f"Checking YOLOE model status in /yoloe_assist: Model loaded = {yoloe_label.yoloe_model is not None}")
    if not yoloe_label.yoloe_model:
        error_msg = "YOLOE AI model is not loaded."
        if yoloe_label.yoloe_model_load_error:
            error_msg += f" Last known error: {yoloe_label.yoloe_model_load_error}"
        print(f"YOLOE model check failed in /yoloe_assist. Error: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 503

    loaded_classes = getattr(yoloe_label, 'yoloe_model_classes', [])
    if not loaded_classes:
        error_msg = "YOLOE model is loaded, but its class list is missing or empty."
        print(f"YOLOE assist check failed: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

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

        predictions = predict_yoloe(image)

        detected_boxes = []
        if predictions:
            print(f"YOLOE raw predictions received: {len(predictions)}")
            for pred in predictions:
                coords = pred.get('coords')
                class_id = pred.get('class_id')

                if not (coords and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
                    print(f"Warning: Skipping invalid coordinate set from YOLOE: {coords}")
                    continue
                if not isinstance(class_id, int) or class_id < 0 or class_id >= len(loaded_classes):
                    print(f"Warning: Skipping prediction with invalid class_id: {class_id} (available: {len(loaded_classes)})")
                    continue

                label_name = loaded_classes[class_id]

                detected_boxes.append({
                    "x_min": int(coords[0]), "y_min": int(coords[1]),
                    "x_max": int(coords[2]), "y_max": int(coords[3]),
                    "label": label_name,
                    "confidence": None
                `

           print(f"Processed {len(detected_boxes)} valid YOLOE bounding boxes.")
        else:
            print("YOLOE Inference returned no valid predictions.")

        print(f"YOLOE Assist finished. Found {len(detected_boxes)} boxes.")
        return jsonify({"success": True, "boxes": detected_boxes})

    except RuntimeError as e:
         print(f"RuntimeError during YOLOE Assist: {e}")
         if "model is not loaded" in str(e).lower():
             yoloe_label.yoloe_model = None
             yoloe_label.yoloe_model_classes = []
         return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        print(f"Error during YOLOE Assist processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error during YOLOE inference: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)  # Enable debug for better error messages`
