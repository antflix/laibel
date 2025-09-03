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
        if not data or 'image_data' not in data:
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
                })

        return jsonify({"success": True, "boxes": detected_boxes})

    except Exception as e:
        print(f"Error processing YOLOE request: {e}")
        return jsonify({"success": False, "error": "Invalid request data"}), 400
