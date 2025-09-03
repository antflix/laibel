                })

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
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)  # Enable debug for better error messages