// venv/laibel/static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
  //
  const MAX_WIDTH = 640;
  const MAX_HEIGHT = 480;
  const handleSize = 8;
  const minBoxSize = 5;

  // --- Global State for Multiple Images ---
  let imageData = []; // Array to hold data for all images { src, filename, originalWidth, originalHeight, scaleRatio, boxes }
  let currentImageIndex = -1; // Index of the currently displayed image
  let image = null; // The actual Image object currently being displayed

  // State related to the *current* image (will be updated on image switch)
  let scaleRatio = 1;
  let originalWidth = 0;
  let originalHeight = 0;
  let currentFilename = "annotated_image";
  let boxes = []; // Holds annotations for the CURRENT image (reference to imageData[currentImageIndex].boxes)

  // Global labels (shared across images)
  let labels = []; // { name: string, color: string }

  // Interaction State
  let currentTool = "draw"; // 'draw' or 'edit'
  let isDrawing = false;
  let isResizing = false;
  let selectedBoxIndex = -1; // Index within the *current* 'boxes' array
  let grabbedHandle = null;
  let startX = 0;
  let startY = 0;

  // --- Model/Prediction States ---
  // YOLO Model State
  let isYoloModelLoaded =
    window.LAIBEL_CONFIG?.yoloModelInitiallyLoaded || false;
  let isYoloLoading = false;
  let yoloModelLoadError = window.LAIBEL_CONFIG?.yoloModelLoadError || null;
  let isYoloPredicting = false; // For YOLO Assist prediction

  // YOLOE Model State
  let isYoloeModelLoaded =
    window.LAIBEL_CONFIG?.yoloeModelInitiallyLoaded || false;
  let isYoloeLoading = false;
  let yoloeModelLoadError = window.LAIBEL_CONFIG?.yoloeModelLoadError || null;
  let isYoloePredicting = false; // For YOLOE Assist prediction
  // Store the classes the YOLOE model was loaded with (if passed from backend)
  let yoloeModelClasses = window.LAIBEL_CONFIG?.yoloeModelClasses || [];

  // Canvas setup
  const canvas = document.getElementById("image-canvas");
  const ctx = canvas.getContext("2d");

  // DOM elements
  const uploadBtn = document.getElementById("upload-btn");
  const imageUpload = document.getElementById("image-upload");
  const saveBtn = document.getElementById("save-btn"); // Export JSON
  const exportYoloBtn = document.getElementById("export-yolo-btn");
  const drawBoxBtn = document.getElementById("draw-box-btn");
  const editBoxBtn = document.getElementById("edit-box-btn");
  const addLabelBtn = document.getElementById("add-label-btn");
  const newLabelInput = document.getElementById("new-label");
  const labelsList = document.getElementById("labels-list");
  const annotationsList = document.getElementById("annotations-list");
  const prevImageBtn = document.getElementById("prev-image-btn");
  const nextImageBtn = document.getElementById("next-image-btn");
  const imageInfoSpan = document.getElementById("image-info");
  const deleteImageBtn = document.getElementById("delete-image-btn");

  // --- Model Buttons ---
  const loadYoloModelBtn = document.getElementById("load-yolo-model-btn");
  const yoloAssistBtn = document.getElementById("yolo-assist-btn");
  const loadYoloeModelBtn = document.getElementById("load-yoloe-model-btn");
  const yoloeAssistBtn = document.getElementById("yoloe-assist-btn");

  // --- Event Listeners ---
  drawBoxBtn.addEventListener("click", () => switchTool("draw"));
  editBoxBtn.addEventListener("click", () => switchTool("edit"));
  uploadBtn.addEventListener("click", () => {
    imageUpload.click();
  });
  prevImageBtn.addEventListener("click", () => {
    if (
      currentImageIndex > 0 &&
      !isYoloPredicting &&
      !isYoloePredicting &&
      !isYoloLoading &&
      !isYoloeLoading
    ) {
      loadImageData(currentImageIndex - 1);
    }
  });
  nextImageBtn.addEventListener("click", () => {
    if (
      currentImageIndex < imageData.length - 1 &&
      !isYoloPredicting &&
      !isYoloePredicting &&
      !isYoloLoading &&
      !isYoloeLoading
    ) {
      loadImageData(currentImageIndex + 1);
    }
  });
  addLabelBtn.addEventListener("click", addNewLabel);
  newLabelInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      addNewLabel();
    }
  });
  saveBtn.addEventListener("click", saveJsonAnnotations); // Export JSON
  exportYoloBtn.addEventListener("click", exportYoloAnnotations);
  deleteImageBtn.addEventListener("click", deleteCurrentImage);

  // Model Button Listeners
  loadYoloModelBtn.addEventListener("click", handleLoadYoloModel);
  yoloAssistBtn.addEventListener("click", handleYoloAssist);
  loadYoloeModelBtn.addEventListener("click", handleLoadYoloeModel); // MODIFIED
  yoloeAssistBtn.addEventListener("click", handleYoloeAssist);

  // --- Keyboard Shortcut Listener ---
  document.addEventListener("keydown", handleKeyDown);

  // --- Tool Switching and State Reset ---
  function switchTool(tool) {
    currentTool = tool;
    if (tool === "draw") {
      drawBoxBtn.classList.add("active");
      editBoxBtn.classList.remove("active");
      canvas.style.cursor = "crosshair";
      resetEditState();
    } else {
      // 'edit'
      editBoxBtn.classList.add("active");
      drawBoxBtn.classList.remove("active");
      canvas.style.cursor = "default";
      resetDrawState();
    }
    redrawCanvas(); // Redraw to show/hide handles
  }

  function resetDrawState() {
    isDrawing = false;
  }

  function resetEditState() {
    isResizing = false;
    selectedBoxIndex = -1;
    grabbedHandle = null;
  }

  // --- Image Upload Handling ---
  imageUpload.addEventListener("change", (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Clear existing data before loading new images
    imageData = [];
    currentImageIndex = -1;
    clearCanvasAndState(); // Clear canvas, reset states, update UI

    const filePromises = [];
    let loadErrors = 0;

    Array.from(files).forEach((file) => {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        const loadPromise = new Promise((resolve, reject) => {
          reader.onload = function (event) {
            const img = new Image();
            img.onload = function () {
              let currentScaleRatio = 1;
              const currentOriginalWidth = img.width;
              const currentOriginalHeight = img.height;

              if (
                currentOriginalWidth > MAX_WIDTH ||
                currentOriginalHeight > MAX_HEIGHT
              ) {
                const widthRatio = MAX_WIDTH / currentOriginalWidth;
                const heightRatio = MAX_HEIGHT / currentOriginalHeight;
                currentScaleRatio = Math.min(widthRatio, heightRatio);
              }

              const data = {
                src: event.target.result, // Base64 data URL
                filename: file.name,
                originalWidth: currentOriginalWidth,
                originalHeight: currentOriginalHeight,
                scaleRatio: currentScaleRatio,
                boxes: [], // Initialize empty boxes for this image
              };
              imageData.push(data);
              resolve(); // Resolve promise for this file
            };
            img.onerror = (err) => {
              console.error(
                "Error loading image into Image object:",
                file.name,
                err,
              );
              loadErrors++;
              reject(new Error(`Failed to load image: ${file.name}`)); // Reject promise
            };
            img.src = event.target.result; // Start loading image object
          };
          reader.onerror = (err) => {
            console.error("Error reading file:", file.name, err);
            loadErrors++;
            reject(new Error(`Failed to read file: ${file.name}`)); // Reject promise
          };
          reader.readAsDataURL(file); // Start reading file
        });
        filePromises.push(loadPromise);
      } else {
        console.warn("Skipping non-image file:", file.name);
      }
    });

    // Wait for all file processing promises
    Promise.allSettled(filePromises) // Use allSettled to continue even if some fail
      .then((results) => {
        const successfulLoads = results.filter(
          (r) => r.status === "fulfilled",
        ).length;
        console.log(
          `Processed ${files.length} files. Successfully loaded ${successfulLoads} images.`,
        );

        if (imageData.length > 0) {
          loadImageData(0); // Load the first successfully processed image
        } else {
          updateNavigationUI(); // Update UI if no images loaded
          if (loadErrors > 0) {
            alert(
              `Failed to load ${loadErrors} image(s). Please check console for details.`,
            );
          }
        }
        // Reset file input value to allow re-uploading the same file(s)
        imageUpload.value = null;
      })
      .catch((error) => {
        // This catch is less likely with Promise.allSettled, but good practice
        console.error("Unexpected error during file processing:", error);
        alert(
          "An unexpected error occurred while loading images. Please check the console.",
        );
        updateNavigationUI();
        imageUpload.value = null;
      });
  });

  // --- Image Loading and Navigation ---
  function loadImageData(index) {
    if (index < 0 || index >= imageData.length) {
      console.error("Invalid image index requested:", index);
      clearCanvasAndState();
      return;
    }

    currentImageIndex = index;
    const data = imageData[currentImageIndex];

    // Update global state from the selected image's data
    originalWidth = data.originalWidth;
    originalHeight = data.originalHeight;
    scaleRatio = data.scaleRatio;
    currentFilename = data.filename;
    boxes = data.boxes; // CRITICAL: Update the 'boxes' reference

    console.log(
      `Loading image ${currentImageIndex + 1}/${imageData.length}: ${currentFilename} (Original: ${originalWidth}x${originalHeight}, Scale: ${scaleRatio.toFixed(3)})`,
    );

    image = new Image();
    image.onload = () => {
      const displayWidth = Math.round(originalWidth * scaleRatio);
      const displayHeight = Math.round(originalHeight * scaleRatio);

      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        console.log(`Canvas resized to: ${displayWidth}x${displayHeight}`);
      }

      resetDrawState();
      resetEditState();
      switchTool(currentTool); // Re-apply current tool cursor etc.
      redrawCanvas();
      updateAnnotationsList(); // Update sidebar list for the new image
      updateNavigationUI(); // Update buttons and image info text
    };
    image.onerror = () => {
      console.error("Error loading image source for display:", data.filename);
      alert(
        `Error loading image: ${data.filename}. It might be corrupted or unsupported.`,
      );
      imageData.splice(currentImageIndex, 1);
      if (imageData.length > 0) {
        loadImageData(Math.max(0, currentImageIndex % imageData.length));
      } else {
        clearCanvasAndState();
      }
    };
    image.src = data.src; // Start loading the image from its base64 source
  }

  function clearCanvasAndState() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    image = null;
    boxes = []; // Clear the reference
    originalWidth = 0;
    originalHeight = 0;
    scaleRatio = 1;
    currentFilename = "annotated_image";
    currentImageIndex = -1; // Indicate no image is loaded
    resetDrawState();
    resetEditState();
    updateAnnotationsList(); // Clear the sidebar list
    updateNavigationUI(); // Update buttons and info text
    console.log("Canvas and state cleared.");
  }

  // Initial setup for model buttons based on backend state passed via window.LAIBEL_CONFIG
  function initializeModelButtons() {
    // YOLO
    if (isYoloModelLoaded) {
      console.log("YOLO Model was already loaded on page initialization.");
    } else if (yoloModelLoadError) {
      console.error("Initial YOLO model state error:", yoloModelLoadError);
    }
    // YOLOE
    if (isYoloeModelLoaded) {
      console.log("YOLOE Model was already loaded on page initialization.");
      console.log("YOLOE loaded classes:", yoloeModelClasses);
    } else if (yoloeModelLoadError) {
      console.error("Initial YOLOE model state error:", yoloeModelLoadError);
    }
    // Call updateNavigationUI to set initial button states correctly
    updateNavigationUI();
  }

  // --- Update Navigation and Button States ---
  function updateNavigationUI() {
    const hasImages = imageData.length > 0;
    const hasCurrentIndex =
      currentImageIndex >= 0 && currentImageIndex < imageData.length;
    const anyLoading = isYoloLoading || isYoloeLoading;
    const anyPredicting = isYoloPredicting || isYoloePredicting;
    const blockActions = anyLoading || anyPredicting; // Block navigation/drawing/editing during loading or prediction

    // Enable/disable delete button
    deleteImageBtn.disabled = !hasCurrentIndex || blockActions;

    // --- Navigation Buttons ---
    prevImageBtn.disabled = currentImageIndex <= 0 || blockActions;
    nextImageBtn.disabled =
      currentImageIndex >= imageData.length - 1 || blockActions;

    // --- Tool Buttons ---
    drawBoxBtn.disabled = blockActions;
    editBoxBtn.disabled = blockActions;

    // --- Image Info ---
    if (!hasImages) {
      imageInfoSpan.textContent = "No images loaded";
    } else {
      if (hasCurrentIndex) {
        const displayFilename =
          imageData[currentImageIndex].filename.length > 25
            ? imageData[currentImageIndex].filename.substring(0, 22) + "..."
            : imageData[currentImageIndex].filename;
        imageInfoSpan.textContent = `${currentImageIndex + 1} / ${imageData.length} (${displayFilename})`;
      } else {
        imageInfoSpan.textContent = `0 / ${imageData.length}`; // Should not happen if hasImages is true
      }
    }

    // --- YOLO Buttons ---
    loadYoloModelBtn.disabled = isYoloModelLoaded || isYoloLoading;
    yoloAssistBtn.disabled =
      !hasCurrentIndex || !isYoloModelLoaded || isYoloPredicting || anyLoading; // Disable if not loaded, predicting, or any model is loading

    if (isYoloLoading) {
      loadYoloModelBtn.textContent = "Loading YOLO...";
    } else if (isYoloModelLoaded) {
      loadYoloModelBtn.textContent = "YOLO Loaded";
    } else {
      loadYoloModelBtn.textContent = "Load YOLO Model";
    }
    if (isYoloPredicting) {
      yoloAssistBtn.textContent = "YOLO Predicting...";
    } else {
      yoloAssistBtn.textContent = "YOLO Assist";
    }

    // --- YOLOE Buttons ---
    const currentLabelNames = labels.map((l) => l.name);
    const requiredLabelsMatchLoaded =
      isYoloeModelLoaded &&
      yoloeModelClasses.length === currentLabelNames.length &&
      yoloeModelClasses.every((label) => currentLabelNames.includes(label)) &&
      currentLabelNames.every((label) => yoloeModelClasses.includes(label));

    // Enable Load button only if model not loaded/loading AND there are labels defined
    loadYoloeModelBtn.disabled =
      isYoloeLoading || requiredLabelsMatchLoaded || labels.length === 0;
    yoloeAssistBtn.disabled =
      !hasCurrentIndex ||
      !isYoloeModelLoaded || // Must be loaded
      !requiredLabelsMatchLoaded || // Must be loaded with current labels
      isYoloePredicting || // Not currently predicting
      anyLoading; // No other model loading

    // Update YOLOE button text
    if (isYoloeLoading) {
      loadYoloeModelBtn.textContent = "Loading YOLOE...";
    } else if (requiredLabelsMatchLoaded) {
      // Show loaded classes if they match
      const displayClasses =
        yoloeModelClasses.length > 2
          ? yoloeModelClasses.slice(0, 2).join(", ") + "..."
          : yoloeModelClasses.join(", ") || "None";
      loadYoloeModelBtn.textContent = `YOLOE Loaded (${displayClasses})`;
    } else if (labels.length === 0) {
      loadYoloeModelBtn.textContent = "Add Labels to Load";
    } else {
      // Needs loading or reloading
      const action = isYoloeModelLoaded ? "Reload" : "Load";
      loadYoloeModelBtn.textContent = `${action} YOLOE Model`;
    }

    if (isYoloePredicting) {
      yoloeAssistBtn.textContent = "YOLOE Predicting...";
    } else {
      yoloeAssistBtn.textContent = "YOLOE Assist";
    }
  }

  // --- Canvas Event Handlers ---
  canvas.addEventListener("mousedown", handleMouseDown);
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener("mouseup", handleMouseUp);
  canvas.addEventListener("mouseleave", handleMouseLeave); // Important for cleanup

  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, canvas.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, canvas.height));
    return { x, y };
  }

  function getHandleUnderMouse(x, y) {
    if (!boxes) return null; // Safety check if boxes array is not ready
    for (let i = boxes.length - 1; i >= 0; i--) {
      const box = boxes[i];
      if (!box) continue; // Skip if box data is somehow invalid
      const hs = handleSize / 2;
      const tl_x = box.x,
        tl_y = box.y;
      const tr_x = box.x + box.width,
        tr_y = box.y;
      const bl_x = box.x,
        bl_y = box.y + box.height;
      const br_x = box.x + box.width,
        br_y = box.y + box.height;

      if (x >= tl_x - hs && x <= tl_x + hs && y >= tl_y - hs && y <= tl_y + hs)
        return { boxIndex: i, handle: "tl" };
      if (x >= tr_x - hs && x <= tr_x + hs && y >= tr_y - hs && y <= tr_y + hs)
        return { boxIndex: i, handle: "tr" };
      if (x >= bl_x - hs && x <= bl_x + hs && y >= bl_y - hs && y <= bl_y + hs)
        return { boxIndex: i, handle: "bl" };
      if (x >= br_x - hs && x <= br_x + hs && y >= br_y - hs && y <= br_y + hs)
        return { boxIndex: i, handle: "br" };
    }
    return null;
  }

  function handleMouseDown(e) {
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (!image || blockActions) return;
    const pos = getMousePos(e);
    startX = pos.x;
    startY = pos.y;

    if (currentTool === "draw") {
      isDrawing = true;
      resetEditState();
    } else if (currentTool === "edit") {
      resetDrawState();
      const handleInfo = getHandleUnderMouse(pos.x, pos.y);
      if (handleInfo) {
        isResizing = true;
        selectedBoxIndex = handleInfo.boxIndex;
        grabbedHandle = handleInfo.handle;
        console.log(
          `Grabbed handle ${grabbedHandle} of box ${selectedBoxIndex}`,
        );
      } else {
        resetEditState();
        redrawCanvas();
      }
    }
  }

  function handleMouseMove(e) {
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (!image || blockActions) return;
    const pos = getMousePos(e);
    const currentX = pos.x;
    const currentY = pos.y;

    if (currentTool === "draw" && isDrawing) {
      redrawCanvas();
      ctx.strokeStyle = "#61afef";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
      ctx.setLineDash([]);
    } else if (currentTool === "edit" && isResizing) {
      if (selectedBoxIndex < 0 || !boxes || !boxes[selectedBoxIndex]) {
        isResizing = false;
        return;
      }
      const box = boxes[selectedBoxIndex];
      const originalBoxX = box.x,
        originalBoxY = box.y;
      const originalBoxWidth = box.width,
        originalBoxHeight = box.height;
      let newX = box.x,
        newY = box.y,
        newWidth = box.width,
        newHeight = box.height;

      switch (grabbedHandle) {
        case "tl":
          newWidth = originalBoxX + originalBoxWidth - currentX;
          newHeight = originalBoxY + originalBoxHeight - currentY;
          newX = currentX;
          newY = currentY;
          break;
        case "tr":
          newWidth = currentX - originalBoxX;
          newHeight = originalBoxY + originalBoxHeight - currentY;
          newY = currentY;
          break;
        case "bl":
          newWidth = originalBoxX + originalBoxWidth - currentX;
          newHeight = currentY - originalBoxY;
          newX = currentX;
          break;
        case "br":
          newWidth = currentX - originalBoxX;
          newHeight = currentY - originalBoxY;
          break;
      }
      box.x = newX;
      box.y = newY;
      box.width = newWidth;
      box.height = newHeight;
      redrawCanvas();
    } else if (currentTool === "edit" && !isResizing) {
      const handleInfo = getHandleUnderMouse(currentX, currentY);
      if (handleInfo) {
        switch (handleInfo.handle) {
          case "tl":
          case "br":
            canvas.style.cursor = "nwse-resize";
            break;
          case "tr":
          case "bl":
            canvas.style.cursor = "nesw-resize";
            break;
          default:
            canvas.style.cursor = "default";
            break;
        }
      } else {
        canvas.style.cursor = "default";
      }
    }
  }

  function handleMouseUp(e) {
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (blockActions) return;

    if (currentTool === "draw" && isDrawing) {
      isDrawing = false;
      const pos = getMousePos(e);
      const endX = pos.x,
        endY = pos.y;
      const finalX = Math.min(startX, endX),
        finalY = Math.min(startY, endY);
      const finalWidth = Math.abs(endX - startX),
        finalHeight = Math.abs(endY - startY);

      if (
        finalWidth >= minBoxSize &&
        finalHeight >= minBoxSize &&
        currentImageIndex !== -1
      ) {
        const newBox = {
          x: finalX,
          y: finalY,
          width: finalWidth,
          height: finalHeight,
          label: labels.length > 0 ? labels[0].name : "unlabeled",
        };
        imageData[currentImageIndex].boxes.push(newBox);
        updateAnnotationsList();
      } else {
        console.log("Box too small or no image, not added.");
      }
      redrawCanvas();
    } else if (currentTool === "edit" && isResizing) {
      if (selectedBoxIndex >= 0 && boxes && boxes[selectedBoxIndex]) {
        const box = boxes[selectedBoxIndex];
        // Normalize box: ensure width/height positive, enforce min size, check boundaries
        if (box.width < 0) {
          box.x += box.width;
          box.width = Math.abs(box.width);
        }
        if (box.height < 0) {
          box.y += box.height;
          box.height = Math.abs(box.height);
        }
        box.width = Math.max(minBoxSize, box.width);
        box.height = Math.max(minBoxSize, box.height);
        box.x = Math.max(0, box.x);
        box.y = Math.max(0, box.y);
        if (box.x + box.width > canvas.width) {
          box.width = canvas.width - box.x;
        }
        if (box.y + box.height > canvas.height) {
          box.height = canvas.height - box.y;
        }
        box.width = Math.max(minBoxSize, box.width); // Re-check min size after clamping
        box.height = Math.max(minBoxSize, box.height);
      }
      resetEditState();
      redrawCanvas();
      updateAnnotationsList();
      canvas.style.cursor = "default";
    }
  }

  function handleMouseLeave(e) {
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (blockActions) return;

    if (isDrawing) {
      isDrawing = false;
      console.log("Drawing cancelled due to mouse leave.");
      redrawCanvas();
    }
    if (isResizing) {
      // Apply same finalization as mouseup
      if (selectedBoxIndex >= 0 && boxes && boxes[selectedBoxIndex]) {
        const box = boxes[selectedBoxIndex];
        if (box.width < 0) {
          box.x += box.width;
          box.width = Math.abs(box.width);
        }
        if (box.height < 0) {
          box.y += box.height;
          box.height = Math.abs(box.height);
        }
        box.width = Math.max(minBoxSize, box.width);
        box.height = Math.max(minBoxSize, box.height);
        box.x = Math.max(0, box.x);
        box.y = Math.max(0, box.y);
        box.width = Math.min(box.width, canvas.width - box.x);
        box.height = Math.min(box.height, canvas.height - box.y);
        box.width = Math.max(minBoxSize, box.width);
        box.height = Math.max(minBoxSize, box.height);
      }
      console.log("Resizing finalized due to mouse leave.");
      resetEditState();
      redrawCanvas();
      updateAnnotationsList();
      canvas.style.cursor = "default";
    } else if (currentTool === "edit") {
      canvas.style.cursor = "default";
    }
  }

  // --- Drawing Canvas ---
  function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const currentImageData =
      currentImageIndex >= 0 ? imageData[currentImageIndex] : null;

    if (image && canvas.width > 0 && canvas.height > 0) {
      try {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      } catch (e) {
        console.error("Error drawing image:", e);
        drawPlaceholder("Error drawing image");
        return;
      }
    } else {
      drawPlaceholder("Upload images to begin");
      return;
    }

    const currentBoxes = currentImageData ? currentImageData.boxes : [];
    if (!currentBoxes) {
      console.warn(
        "RedrawCanvas: No boxes array found for current image index:",
        currentImageIndex,
      );
      return;
    }

    currentBoxes.forEach((box, index) => {
      const labelObj = labels.find((l) => l.name === box.label);
      const color = labelObj ? labelObj.color : "#CCCCCC"; // Default grey for unknown labels
      const fontColor = getContrastYIQ(color);

      if (
        isNaN(box.x) ||
        isNaN(box.y) ||
        isNaN(box.width) ||
        isNaN(box.height)
      ) {
        console.warn(
          `RedrawCanvas: Skipping box index ${index} due to invalid coordinates:`,
          box,
        );
        return;
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Draw label text background and text
      const text = box.label || "unlabeled"; // Ensure there's always text
      ctx.font = "bold 12px Arial";
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const textHeight = 16; // Approximate height for background
      const textPad = 5;

      let bgY = box.y - textHeight; // Position above the box
      // Adjust if the label goes off the top of the canvas
      if (bgY < 0) {
        bgY = box.y + 2; // Position slightly inside the box top
      }

      ctx.fillStyle = color; // Background color for the label
      ctx.fillRect(box.x, bgY, textWidth + textPad * 2, textHeight);

      ctx.fillStyle = fontColor; // Text color
      ctx.fillText(text, box.x + textPad, bgY + textHeight - 4); // Adjust Y for vertical alignment

      // Draw resize handles if in edit mode
      if (currentTool === "edit") {
        ctx.fillStyle = color;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;
        const hs = handleSize / 2;
        // Draw filled rectangle and then stroke for outline
        ctx.fillRect(box.x - hs, box.y - hs, handleSize, handleSize); // TL fill
        ctx.strokeRect(box.x - hs, box.y - hs, handleSize, handleSize); // TL stroke

        ctx.fillRect(
          box.x + box.width - hs,
          box.y - hs,
          handleSize,
          handleSize,
        ); // TR fill
        ctx.strokeRect(
          box.x + box.width - hs,
          box.y - hs,
          handleSize,
          handleSize,
        ); // TR stroke

        ctx.fillRect(
          box.x - hs,
          box.y + box.height - hs,
          handleSize,
          handleSize,
        ); // BL fill
        ctx.strokeRect(
          box.x - hs,
          box.y + box.height - hs,
          handleSize,
          handleSize,
        ); // BL stroke

        ctx.fillRect(
          box.x + box.width - hs,
          box.y + box.height - hs,
          handleSize,
          handleSize,
        ); // BR fill
        ctx.strokeRect(
          box.x + box.width - hs,
          box.y + box.height - hs,
          handleSize,
          handleSize,
        ); // BR stroke
      }
    });
  }

  function drawPlaceholder(text) {
    ctx.fillStyle = "#33373e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#828a9a";
    ctx.font = "16px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  }

  // --- Label Management ---
  function addNewLabel() {
    const labelName = newLabelInput.value.trim();
    if (!labelName) {
      alert("Please enter a label name.");
      return;
    }
    if (labels.some((l) => l.name.toLowerCase() === labelName.toLowerCase())) {
      alert(`Label "${labelName}" already exists (case-insensitive).`);
      return;
    }
    const color = getRandomColor();
    labels.push({ name: labelName, color: color });
    updateLabelsList();
    updateAnnotationsList(); // Update dropdowns
    updateNavigationUI(); // Update YOLOE button states based on labels
    newLabelInput.value = "";
    newLabelInput.focus();

    // Auto-assign the first label to existing 'unlabeled' boxes?
    // if (labels.length === 1 && currentImageIndex !== -1) {
    //     let changed = false;
    //     imageData[currentImageIndex].boxes.forEach((box) => {
    //         if (box.label === "unlabeled") {
    //             box.label = labelName;
    //             changed = true;
    //         }
    //     });
    //     if (changed) {
    //         redrawCanvas();
    //         updateAnnotationsList();
    //     }
    // }
    console.log(`Added label: ${labelName} (${color})`);
  }

  function updateLabelsList() {
    labelsList.innerHTML = "";
    if (labels.length === 0) {
      labelsList.innerHTML =
        '<p style="color: var(--text-secondary); font-style: italic; padding: 5px;">No labels defined yet.</p>';
    } else {
      labels.forEach((label, index) => {
        const labelItem = document.createElement("div");
        labelItem.className = "label-item";
        const labelDisplay = document.createElement("div");
        labelDisplay.innerHTML = `<span class="label-color" style="background-color: ${label.color}"></span>${escapeHtml(label.name)}`;
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn delete-label-btn";
        deleteBtn.textContent = "✕";
        deleteBtn.title = `Delete label "${escapeHtml(label.name)}"`;
        deleteBtn.onclick = (e) => {
          e.stopPropagation();
          deleteLabel(index);
        };
        labelItem.appendChild(labelDisplay);
        labelItem.appendChild(deleteBtn);
        labelsList.appendChild(labelItem);
      });
    }
  }

  function deleteLabel(indexToDelete) {
    const labelToDelete = labels[indexToDelete];
    if (!labelToDelete) return;
    const labelNameToDelete = labelToDelete.name;
    if (
      !confirm(
        `Are you sure you want to delete the label "${labelNameToDelete}"? \nThis will change annotations using this label to 'unlabeled'.`,
      )
    ) {
      return;
    }

    console.log(`Deleting label: ${labelNameToDelete}`);
    labels.splice(indexToDelete, 1);
    const fallbackLabel = "unlabeled"; // Always fallback to unlabeled

    imageData.forEach((imgData, imgIndex) => {
      let changed = false;
      imgData.boxes.forEach((box) => {
        if (box.label === labelNameToDelete) {
          box.label = fallbackLabel;
          changed = true;
        }
      });
      if (changed && imgIndex === currentImageIndex) {
        // Only redraw if the current image was affected
        redrawCanvas();
        updateAnnotationsList(); // Update sidebar immediately
      } else if (changed) {
        console.log(`Updated labels in background image: ${imgData.filename}`);
      }
    });

    updateLabelsList(); // Update label list display
    updateNavigationUI(); // Update YOLOE buttons status

    // Update annotation list dropdowns even if current image not redrawn,
    // as available options changed.
    if (currentImageIndex !== -1) {
      updateAnnotationsList();
    }
  }

  // --- Annotation List Management ---
  function updateAnnotationsList() {
    annotationsList.innerHTML = "";
    if (
      currentImageIndex === -1 ||
      !imageData[currentImageIndex] ||
      imageData[currentImageIndex].boxes.length === 0
    ) {
      annotationsList.innerHTML =
        '<p style="color: var(--text-secondary); font-style: italic; padding: 5px;">No annotations for this image.</p>';
      // Ensure boxes reference is cleared/reset if needed, though handled by loadImageData
      boxes =
        currentImageIndex !== -1 ? imageData[currentImageIndex].boxes : [];
      return;
    }

    boxes = imageData[currentImageIndex].boxes; // Ensure 'boxes' points to the correct array

    boxes.forEach((box, index) => {
      const annotationItem = document.createElement("div");
      annotationItem.className = "annotation-item";
      const labelSelect = document.createElement("select");
      labelSelect.className = "annotation-label-select";
      labelSelect.title = `Label for box ${index + 1}`;

      // Ensure 'unlabeled' option is always present and selected if needed
      const unlabeledOption = document.createElement("option");
      unlabeledOption.value = "unlabeled";
      unlabeledOption.textContent = "unlabeled";
      let currentLabelExistsInList = labels.some((l) => l.name === box.label);
      unlabeledOption.selected =
        !currentLabelExistsInList || box.label === "unlabeled";
      labelSelect.appendChild(unlabeledOption);

      // Add defined labels
      labels.forEach((label) => {
        const option = document.createElement("option");
        option.value = label.name;
        option.textContent = escapeHtml(label.name);
        if (box.label === label.name) {
          option.selected = true;
        }
        labelSelect.appendChild(option);
      });

      labelSelect.onchange = (e) => {
        const newLabel = e.target.value;
        imageData[currentImageIndex].boxes[index].label = newLabel;
        console.log(`Box ${index} label changed to: ${newLabel}`);
        redrawCanvas(); // Redraw canvas to show new label color/text
      };

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "btn delete-annotation-btn";
      deleteBtn.textContent = "✕";
      deleteBtn.title = `Delete annotation ${index + 1}`;
      deleteBtn.onclick = () => {
        deleteAnnotation(index);
      };

      annotationItem.appendChild(labelSelect);
      annotationItem.appendChild(deleteBtn);
      annotationsList.appendChild(annotationItem);
    });
  }

  function deleteAnnotation(indexToDelete) {
    if (currentImageIndex === -1 || !imageData[currentImageIndex]) return;

    // If currently resizing the box being deleted, reset edit state
    if (isResizing && selectedBoxIndex === indexToDelete) {
      resetEditState();
      canvas.style.cursor = currentTool === "edit" ? "default" : "crosshair"; // Reset cursor
    }

    imageData[currentImageIndex].boxes.splice(indexToDelete, 1);
    console.log(`Deleted annotation ${indexToDelete + 1}`);

    // Adjust selectedBoxIndex if it's affected by the deletion
    if (selectedBoxIndex === indexToDelete) {
      resetEditState(); // The selected box is gone
    } else if (selectedBoxIndex > indexToDelete) {
      selectedBoxIndex--; // Adjust index for subsequent boxes
    }

    updateAnnotationsList(); // Update the sidebar
    redrawCanvas(); // Redraw the canvas without the deleted box
  }

  // --- Function to handle loading the YOLO model ---
  async function handleLoadYoloModel() {
    if (isYoloModelLoaded || isYoloLoading) {
      console.log("YOLO Model already loaded or loading in progress.");
      return;
    }
    console.log("Requesting YOLO model load...");
    isYoloLoading = true;
    updateNavigationUI();

    try {
      const response = await fetch("/load_yolo_model", { method: "POST" });
      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch (jsonError) {
          errorMsg += `: ${response.statusText || "Failed to load YOLO model"}`;
        }
        throw new Error(errorMsg);
      }
      const result = await response.json();
      if (result.success) {
        console.log("YOLO Model loaded successfully:", result.message || "");
        isYoloModelLoaded = true;
        yoloModelLoadError = null;
        alert("YOLO Model loaded successfully!");
      } else {
        throw new Error(
          result.error || "Failed to load YOLO model. Unknown error.",
        );
      }
    } catch (error) {
      console.error("Failed to load YOLO model:", error);
      isYoloModelLoaded = false;
      yoloModelLoadError = error.message;
      alert(`Error loading YOLO Model: ${error.message}`);
    } finally {
      isYoloLoading = false;
      updateNavigationUI();
    }
  }

  // --- Function to handle loading the YOLOE model (MODIFIED) ---
  async function handleLoadYoloeModel() {
    // Get current label names
    const currentLabelNames = labels.map((l) => l.name);

    if (currentLabelNames.length === 0) {
      alert("Please define at least one label before loading the YOLOE model.");
      return;
    }

    // Check if already loading
    if (isYoloeLoading) {
      console.log("YOLOE model loading already in progress.");
      return;
    }

    // Check if model is loaded AND classes match exactly
    const requiredLabelsMatchLoaded =
      isYoloeModelLoaded &&
      yoloeModelClasses.length === currentLabelNames.length &&
      yoloeModelClasses.every((label) => currentLabelNames.includes(label)) &&
      currentLabelNames.every((label) => yoloeModelClasses.includes(label));

    if (requiredLabelsMatchLoaded) {
      console.log("YOLOE model already loaded with the current set of labels.");
      alert("YOLOE model already loaded with the current labels.");
      return;
    }

    const action = isYoloeModelLoaded ? "Reloading" : "Requesting";
    console.log(`${action} YOLOE model load with labels:`, currentLabelNames);
    isYoloeLoading = true;
    updateNavigationUI();

    try {
      const response = await fetch("/load_yoloe_model", {
        method: "POST",
        // Send the labels in the request body
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ labels: currentLabelNames }),
      });

      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch (jsonError) {
          errorMsg += `: ${response.statusText || "Failed to load YOLOE model"}`;
        }
        throw new Error(errorMsg);
      }
      const result = await response.json();
      if (result.success) {
        console.log(
          "YOLOE Model loaded/reloaded successfully:",
          result.message || "",
        );
        isYoloeModelLoaded = true;
        yoloeModelLoadError = null;
        // Update the locally stored classes based on success response
        yoloeModelClasses = [...currentLabelNames]; // Store the labels used for loading
        alert(result.message || "YOLOE Model loaded successfully!");
      } else {
        throw new Error(
          result.error || "Failed to load YOLOE model. Unknown error.",
        );
      }
    } catch (error) {
      console.error("Failed to load YOLOE model:", error);
      // Don't assume model is loaded on error
      // isYoloeModelLoaded = false; // Keep previous state? Or reset? Resetting seems safer.
      // yoloeModelClasses = [];
      yoloeModelLoadError = error.message;
      alert(`Error loading YOLOE Model: ${error.message}`);
      // If loading failed, the model might be in an unusable state or still loaded with old classes
      // Best to reflect that it's *not* ready with the current labels
      isYoloeModelLoaded = false; // Mark as not correctly loaded for current context
      yoloeModelClasses = [];
    } finally {
      isYoloeLoading = false;
      updateNavigationUI(); // Update button states reflecting success/failure
    }
  }

  // --- Function to handle YOLO AI Assist ---
  async function handleYoloAssist() {
    if (!isYoloModelLoaded) {
      alert("YOLO Model is not loaded. Please click 'Load YOLO Model' first.");
      return;
    }
    if (
      currentImageIndex === -1 ||
      !imageData[currentImageIndex] ||
      isYoloPredicting ||
      isYoloePredicting ||
      isYoloLoading ||
      isYoloeLoading
    ) {
      console.log(
        "YOLO Assist cannot run: No image, operation in progress, or model not ready.",
      );
      return;
    }

    const currentImageData = imageData[currentImageIndex];
    console.log(`Requesting YOLO prediction for: ${currentImageData.filename}`);
    isYoloPredicting = true;
    updateNavigationUI();
    canvas.style.opacity = "0.7";
    canvas.style.cursor = "wait";

    try {
      const response = await fetch("/ai_assist", {
        // Endpoint for YOLO
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: currentImageData.src }),
      });
      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch (e) {
          /* Ignore */
        }
        if (response.status === 503) {
          // Model not loaded server-side?
          errorMsg +=
            ". The YOLO model might not be loaded on the server. Try loading it again.";
          isYoloModelLoaded = false; // Reset frontend state
        }
        throw new Error(errorMsg);
      }
      const result = await response.json();
      if (result.success && result.boxes) {
        console.log(
          "YOLO prediction successful. Received boxes:",
          result.boxes.length,
        );
        addPredictionsToCanvas(result.boxes, "YOLO"); // Pass type for logging
      } else {
        throw new Error(
          result.error ||
            "YOLO Prediction failed: No boxes found or backend error.",
        );
      }
    } catch (error) {
      console.error("YOLO Assist failed:", error);
      alert(`YOLO Assist Error: ${error.message}`);
    } finally {
      isYoloPredicting = false;
      updateNavigationUI();
      canvas.style.opacity = "1";
      canvas.style.cursor = currentTool === "draw" ? "crosshair" : "default";
    }
  }

  // --- Function to handle YOLOE AI Assist ---
  async function handleYoloeAssist() {
    // Check if model is loaded and classes match
    const currentLabelNames = labels.map((l) => l.name);
    const requiredLabelsMatchLoaded =
      isYoloeModelLoaded &&
      yoloeModelClasses.length === currentLabelNames.length &&
      yoloeModelClasses.every((label) => currentLabelNames.includes(label)) &&
      currentLabelNames.every((label) => yoloeModelClasses.includes(label));

    if (!requiredLabelsMatchLoaded) {
      alert(
        "YOLOE Model is not loaded or not loaded with the current set of labels. Please click 'Load/Reload YOLOE Model'.",
      );
      return;
    }

    if (
      currentImageIndex === -1 ||
      !imageData[currentImageIndex] ||
      isYoloPredicting ||
      isYoloePredicting ||
      isYoloLoading ||
      isYoloeLoading
    ) {
      console.log(
        "YOLOE Assist cannot run: No image, operation in progress, or model not ready.",
      );
      return;
    }

    const currentImageData = imageData[currentImageIndex];
    console.log(
      `Requesting YOLOE prediction for: ${currentImageData.filename} (using classes: ${yoloeModelClasses.join(", ")})`,
    );
    isYoloePredicting = true;
    updateNavigationUI();
    canvas.style.opacity = "0.7";
    canvas.style.cursor = "wait";

    try {
      const response = await fetch("/yoloe_assist", {
        // Endpoint for YOLOE
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: currentImageData.src }),
      });
      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.error || errorMsg;
        } catch (e) {
          /* Ignore */
        }
        if (response.status === 503) {
          // Model not loaded server-side?
          errorMsg +=
            ". The YOLOE model might not be loaded on the server. Try loading it again.";
          // Reset frontend state as backend state is likely wrong
          isYoloeModelLoaded = false;
          yoloeModelClasses = [];
        } else if (
          response.status === 500 &&
          errorMsg.includes("class list is missing")
        ) {
          // Specific error if classes mismatch server-side somehow
          isYoloeModelLoaded = false; // Consider it not loaded correctly
          yoloeModelClasses = [];
        }
        throw new Error(errorMsg);
      }
      const result = await response.json();
      if (result.success && result.boxes) {
        console.log(
          "YOLOE prediction successful. Received boxes:",
          result.boxes.length,
        );
        addPredictionsToCanvas(result.boxes, "YOLOE"); // Pass type for logging
      } else {
        // Handle specific backend errors if possible
        if (result.error && result.error.includes("model is not loaded")) {
          isYoloeModelLoaded = false;
          yoloeModelClasses = [];
        }
        throw new Error(
          result.error ||
            "YOLOE Prediction failed: No boxes found or backend error.",
        );
      }
    } catch (error) {
      console.error("YOLOE Assist failed:", error);
      alert(`YOLOE Assist Error: ${error.message}`);
    } finally {
      isYoloePredicting = false;
      updateNavigationUI(); // Update UI reflecting potential state changes on error
      canvas.style.opacity = "1";
      canvas.style.cursor = currentTool === "draw" ? "crosshair" : "default";
    }
  }

  function addPredictionsToCanvas(predictions, modelType = "AI") {
    if (currentImageIndex === -1 || !imageData[currentImageIndex]) return;

    const currentImageData = imageData[currentImageIndex];
    const currentScaleRatio = currentImageData.scaleRatio;
    let boxesAdded = 0;

    predictions.forEach((pred) => {
      // Ensure coordinates are valid numbers
      const originalX = Number(pred.x_min);
      const originalY = Number(pred.y_min);
      const originalXMax = Number(pred.x_max);
      const originalYMax = Number(pred.y_max);

      if (
        isNaN(originalX) ||
        isNaN(originalY) ||
        isNaN(originalXMax) ||
        isNaN(originalYMax)
      ) {
        console.warn(
          `Skipping ${modelType} predicted box due to non-numeric coordinates:`,
          pred,
        );
        return;
      }

      const originalWidth = originalXMax - originalX;
      const originalHeight = originalYMax - originalY;

      if (originalWidth <= 0 || originalHeight <= 0) {
        console.warn(
          `Skipping ${modelType} predicted box with zero or negative dimensions:`,
          pred,
        );
        return;
      }

      const canvasX = Math.round(originalX * currentScaleRatio);
      const canvasY = Math.round(originalY * currentScaleRatio);
      const canvasWidth = Math.round(originalWidth * currentScaleRatio);
      const canvasHeight = Math.round(originalHeight * currentScaleRatio);

      if (canvasWidth < minBoxSize || canvasHeight < minBoxSize) {
        console.warn(
          `Skipping ${modelType} predicted box for label '${pred.label}' - too small on canvas (${canvasWidth}x${canvasHeight})`,
        );
        return;
      }

      const labelName = pred.label || "unlabeled"; // Default label if missing

      // Check if the predicted label exists in our defined labels.
      // If not, we still add it, but log a message.
      if (!labels.some((l) => l.name === labelName)) {
        console.log(
          `${modelType} predicted label "${labelName}" which is not currently defined in the UI. Adding box with this label. Consider adding "${labelName}" to the labels list.`,
        );
        // You could optionally auto-add it here:
        // if (!labels.some(l => l.name === labelName)) {
        //     const color = getRandomColor();
        //     labels.push({ name: labelName, color: color });
        //     updateLabelsList(); // Update UI
        //     updateNavigationUI(); // Update model buttons
        // }
      }

      const newBox = {
        x: Math.max(0, canvasX), // Clamp coordinates to canvas bounds
        y: Math.max(0, canvasY),
        width: Math.min(canvasWidth, canvas.width - Math.max(0, canvasX)),
        height: Math.min(canvasHeight, canvas.height - Math.max(0, canvasY)),
        label: labelName,
        confidence: pred.confidence, // Store confidence if provided
      };

      // Final check on clamped dimensions
      if (newBox.width < minBoxSize || newBox.height < minBoxSize) {
        console.warn(
          `Skipping ${modelType} predicted box for label '${labelName}' after clamping - too small.`,
        );
        return;
      }

      currentImageData.boxes.push(newBox);
      boxesAdded++;
    });

    if (boxesAdded > 0) {
      console.log(
        `Added ${boxesAdded} ${modelType} predicted boxes to the canvas.`,
      );
      redrawCanvas();
      updateAnnotationsList(); // Update the sidebar list
      // Optional: Give user feedback
      // alert(`${boxesAdded} boxes added by ${modelType} Assist.`);
    } else {
      console.log(
        `No valid boxes were added from the ${modelType} prediction results.`,
      );
      // Optional: Give user feedback
      // alert(`${modelType} Assist finished, but no new boxes met the criteria.`);
    }
  }

  // --- Utilities ---
  function getRandomColor() {
    const letters = "0123456789ABCDEF";
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    // Check brightness (simple check)
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    // Avoid very light or very dark colors for better visibility against text/handles
    if (brightness > 220 || brightness < 40) {
      return getRandomColor(); // Retry
    }
    return color;
  }

  function getContrastYIQ(hexcolor) {
    if (!hexcolor || typeof hexcolor !== "string") return "#000000"; // Default black for invalid input
    if (hexcolor.startsWith("#")) {
      hexcolor = hexcolor.slice(1);
    }
    if (hexcolor.length !== 6) return "#000000"; // Default black for invalid length

    const r = parseInt(hexcolor.substr(0, 2), 16);
    const g = parseInt(hexcolor.substr(2, 2), 16);
    const b = parseInt(hexcolor.substr(4, 2), 16);
    if (isNaN(r) || isNaN(g) || isNaN(b)) return "#000000"; // Default black if parsing failed

    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 128 ? "#000000" : "#FFFFFF"; // Black text on light colors, White text on dark colors
  }

  function escapeHtml(unsafe) {
    if (!unsafe) return "";
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function downloadContent(content, filename, mimeType = "application/json") {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }

  // --- Save JSON Annotations Function ---
  function saveJsonAnnotations() {
    if (imageData.length === 0) {
      alert("No images loaded to export annotations for.");
      return;
    }
    const hasAnnotations = imageData.some(
      (imgData) => imgData.boxes && imgData.boxes.length > 0,
    );
    if (!hasAnnotations) {
      if (
        !confirm(
          "No annotations have been made. Export empty structure anyway?",
        )
      ) {
        return;
      }
    }

    const allAnnotations = {
      // Include defined labels in the export
      labels: labels.map((l) => ({ name: l.name, color: l.color })),
      annotations_by_image: imageData.map((imgData) => ({
        image_filename: imgData.filename,
        image_width: imgData.originalWidth,
        image_height: imgData.originalHeight,
        boxes: imgData.boxes
          .map((box) => {
            const scale = imgData.scaleRatio;
            // Handle potential division by zero or invalid scale
            const safeScale = scale > 0 && isFinite(scale) ? scale : 1;

            // Calculate original coordinates
            let x_min = box.x / safeScale;
            let y_min = box.y / safeScale;
            let x_max = (box.x + box.width) / safeScale;
            let y_max = (box.y + box.height) / safeScale;

            // Round to nearest integer
            x_min = Math.round(x_min);
            y_min = Math.round(y_min);
            x_max = Math.round(x_max);
            y_max = Math.round(y_max);

            // Clamp coordinates strictly within image bounds [0, width/height]
            const clamped_x_min = Math.max(
              0,
              Math.min(x_min, imgData.originalWidth),
            );
            const clamped_y_min = Math.max(
              0,
              Math.min(y_min, imgData.originalHeight),
            );
            // Ensure max is at least min, and within bounds
            const clamped_x_max = Math.max(
              clamped_x_min,
              Math.min(x_max, imgData.originalWidth),
            );
            const clamped_y_max = Math.max(
              clamped_y_min,
              Math.min(y_max, imgData.originalHeight),
            );

            // Final check: width/height must be > 0 after clamping
            if (
              clamped_x_max <= clamped_x_min ||
              clamped_y_max <= clamped_y_min
            ) {
              console.warn(
                `Filtering out zero-area box after clamping for image ${imgData.filename}:`,
                box,
              );
              return null; // Mark for filtering
            }

            return {
              x_min: clamped_x_min,
              y_min: clamped_y_min,
              x_max: clamped_x_max,
              y_max: clamped_y_max,
              label: box.label || "unlabeled", // Ensure label exists
            };
          })
          .filter((b) => b !== null), // Remove null entries (filtered boxes)
      })),
    };
    const jsonStr = JSON.stringify(allAnnotations, null, 2);
    console.log("Exporting all annotations as JSON.");
    downloadContent(
      jsonStr,
      `laibel_annotations_${Date.now()}.json`, // Changed filename slightly
      "application/json",
    );
  }

  // --- Export YOLO Annotations Function ---
  function exportYoloAnnotations() {
    if (labels.length === 0) {
      alert("Define labels before exporting in YOLO format.");
      return;
    }
    if (imageData.length === 0) {
      alert("No images loaded to export annotations for.");
      return;
    }

    const labelIndexMap = new Map(
      labels.map((label, index) => [label.name, index]),
    );
    console.log("Label Map for YOLO Export:", labelIndexMap);
    let exportedFiles = 0,
      skippedImages = 0,
      totalSkippedBoxes = 0;

    imageData.forEach((imgData) => {
      // Skip images with no boxes early
      if (!imgData.boxes || imgData.boxes.length === 0) {
        // Don't count this as skipped if it never had annotations
        // skippedImages++;
        return;
      }

      let yoloContent = "";
      let skippedBoxesInImage = 0;
      let validBoxesInImage = 0;

      imgData.boxes.forEach((box) => {
        const labelIndex = labelIndexMap.get(box.label);
        if (labelIndex === undefined) {
          console.warn(
            `Skipping box: Label "${box.label}" not found in defined labels for image ${imgData.filename}.`,
          );
          skippedBoxesInImage++;
          return;
        }

        // Validate image dimensions
        if (imgData.originalWidth <= 0 || imgData.originalHeight <= 0) {
          console.error(
            `Skipping box: Invalid image dimensions (${imgData.originalWidth}x${imgData.originalHeight}) for image ${imgData.filename}.`,
          );
          skippedBoxesInImage++;
          return;
        }

        const scale = imgData.scaleRatio;
        const safeScale = scale > 0 && isFinite(scale) ? scale : 1;

        // Calculate original coordinates (center x, center y, width, height)
        const original_x = box.x / safeScale;
        const original_y = box.y / safeScale;
        const original_box_width = box.width / safeScale;
        const original_box_height = box.height / safeScale;

        const original_x_center = original_x + original_box_width / 2;
        const original_y_center = original_y + original_box_height / 2;

        // Normalize coordinates [0.0, 1.0]
        const norm_x_center = original_x_center / imgData.originalWidth;
        const norm_y_center = original_y_center / imgData.originalHeight;
        const norm_width = original_box_width / imgData.originalWidth;
        const norm_height = original_box_height / imgData.originalHeight;

        // Clamp normalized values to [0.0, 1.0]
        const clamp = (val) => Math.max(0.0, Math.min(1.0, val));
        const clamped_norm_x_center = clamp(norm_x_center);
        const clamped_norm_y_center = clamp(norm_y_center);
        const clamped_norm_width = clamp(norm_width);
        const clamped_norm_height = clamp(norm_height);

        // Validate calculated values and dimensions before adding
        if (
          isNaN(clamped_norm_x_center) ||
          isNaN(clamped_norm_y_center) ||
          isNaN(clamped_norm_width) ||
          isNaN(clamped_norm_height) ||
          clamped_norm_width <= 0 || // Width and height must be > 0
          clamped_norm_height <= 0
        ) {
          console.error(
            `Invalid YOLO calculation for box (label: ${box.label}) in image ${imgData.filename}. Skipping. Values: xc=${clamped_norm_x_center}, yc=${clamped_norm_y_center}, w=${clamped_norm_width}, h=${clamped_norm_height}`,
          );
          skippedBoxesInImage++;
          return;
        }
        yoloContent += `${labelIndex} ${clamped_norm_x_center.toFixed(6)} ${clamped_norm_y_center.toFixed(6)} ${clamped_norm_width.toFixed(6)} ${clamped_norm_height.toFixed(6)}\n`;
        validBoxesInImage++;
      });

      totalSkippedBoxes += skippedBoxesInImage;

      // Only download if there was at least one valid box written to the content
      if (validBoxesInImage > 0) {
        const baseFilename =
          imgData.filename.substring(0, imgData.filename.lastIndexOf(".")) ||
          imgData.filename;
        const yoloFilename = `${baseFilename}.txt`;
        downloadContent(yoloContent, yoloFilename, "text/plain");
        exportedFiles++;
      } else if (imgData.boxes.length > 0) {
        // Log if an image had boxes, but none were valid for export
        console.warn(
          `No valid YOLO annotations generated for image ${imgData.filename} (all ${imgData.boxes.length} boxes skipped or invalid).`,
        );
        skippedImages++;
      }
    });

    // --- Report Results ---
    let message = "";
    if (exportedFiles > 0) {
      message += `${exportedFiles} YOLO annotation file(s) prepared for download.\n`;
    }
    if (skippedImages > 0) {
      message += `${skippedImages} image(s) had annotations, but none were valid for YOLO export.\n`;
    }
    if (totalSkippedBoxes > 0) {
      message += `${totalSkippedBoxes} individual box(es) were skipped due to missing labels or calculation errors.\n`;
    }

    if (message) {
      alert(message.trim() + "\nCheck console for details.");
    } else if (
      imageData.length > 0 &&
      !imageData.some((d) => d.boxes?.length > 0)
    ) {
      alert("No annotations found across all images to export in YOLO format.");
    } else if (imageData.length === 0) {
      // Already handled at the start
    } else {
      // This case means images exist, annotations exist, but somehow nothing was exported or skipped
      alert(
        "YOLO export finished. No files were generated. Check console for potential issues.",
      );
    }
  }

  // --- Function to Delete Current Image ---
  function deleteCurrentImage() {
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (
      currentImageIndex < 0 ||
      currentImageIndex >= imageData.length ||
      blockActions
    ) {
      console.warn(
        "Delete button clicked but no valid image selected or operation in progress.",
      );
      return;
    }
    const imageToDelete = imageData[currentImageIndex];
    if (
      !confirm(
        `Are you sure you want to delete image "${imageToDelete.filename}"? \nAll its annotations will be lost.`,
      )
    ) {
      return;
    }

    console.log(
      `Deleting image index ${currentImageIndex}: ${imageToDelete.filename}`,
    );
    imageData.splice(currentImageIndex, 1);
    let nextIndexToLoad = -1;

    if (imageData.length === 0) {
      // No images left
      nextIndexToLoad = -1;
      console.log("Last image deleted.");
    } else if (currentImageIndex >= imageData.length) {
      // If the last image was deleted, load the new last image
      nextIndexToLoad = imageData.length - 1;
    } else {
      // Otherwise, load the image that shifted into the current index
      nextIndexToLoad = currentImageIndex;
    }

    // Perform the load/clear action
    if (nextIndexToLoad === -1) {
      clearCanvasAndState();
    } else {
      // Temporarily set index to -1 to ensure loadImageData correctly updates state
      const targetIndex = nextIndexToLoad;
      currentImageIndex = -1;
      loadImageData(targetIndex);
    }
    // Update UI after loading/clearing is done
    updateNavigationUI();
  }

  // --- Keyboard Shortcut Handling ---
  function handleKeyDown(event) {
    const activeElement = document.activeElement;
    const isInputFocused =
      activeElement &&
      (activeElement.tagName === "INPUT" ||
        activeElement.tagName === "TEXTAREA" ||
        activeElement.tagName === "SELECT");

    // Ignore if typing in an input/select or if modifier keys (Ctrl, Alt, Meta) are pressed
    if (isInputFocused || event.ctrlKey || event.altKey || event.metaKey) {
      return;
    }

    // Ignore if any model operation is in progress
    const blockActions =
      isYoloLoading || isYoloeLoading || isYoloPredicting || isYoloePredicting;
    if (blockActions) {
      console.log("Keydown ignored: Operation in progress.");
      return;
    }

    // Prevent default browser behavior for handled keys
    let preventDefault = true;

    switch (event.key.toLowerCase()) {
      case "f": // Next image ('f' or ArrowRight)
      case "arrowright":
        if (!nextImageBtn.disabled) {
          nextImageBtn.click();
        }
        break;
      case "d": // Previous image ('d' or ArrowLeft) - Changed from 'r'
      case "arrowleft":
        if (!prevImageBtn.disabled) {
          prevImageBtn.click();
        }
        break;
      case "w": // Draw tool ('w') - Changed from 'd'
        if (!drawBoxBtn.disabled) {
          switchTool("draw");
        }
        break;
      case "e": // Edit tool ('e')
        if (!editBoxBtn.disabled) {
          switchTool("edit");
        }
        break;
      case "q": // Load/Reload YOLOE Model ('q') - Changed from unused
        if (!loadYoloeModelBtn.disabled) {
          loadYoloeModelBtn.click();
        }
        break;
      case "a": // YOLOE Assist ('a') - Changed from YOLO Assist
        if (!yoloeAssistBtn.disabled) {
          yoloeAssistBtn.click();
        }
        break;
      case "s": // Load YOLO Model ('s') - Changed from unused
        if (!loadYoloModelBtn.disabled) {
          loadYoloModelBtn.click();
        }
        break;
      case "z": // YOLO Assist ('z') - Changed from unused
        if (!yoloAssistBtn.disabled) {
          yoloAssistBtn.click();
        }
        break;
      case "delete":
      case "backspace":
        // Delete selected annotation if in edit mode and a box is selected via handle grab
        if (currentTool === "edit" && selectedBoxIndex !== -1) {
          console.log(
            `Delete shortcut: Deleting annotation ${selectedBoxIndex + 1}`,
          );
          deleteAnnotation(selectedBoxIndex);
          // Note: resetEditState happens inside deleteAnnotation if needed
        } else {
          console.log(
            "Delete shortcut pressed, but no annotation selected in edit mode.",
          );
          preventDefault = false; // Don't prevent default if not deleting
        }
        break;
      default:
        preventDefault = false; // Don't prevent default for unhandled keys
        break;
    }

    if (preventDefault) {
      event.preventDefault();
    }
  }

  // --- Initial Setup on Page Load ---
  function initializeApp() {
    console.log("Initializing Laibel Application...");
    updateLabelsList();
    updateAnnotationsList();
    initializeModelButtons(); // Sets up initial button states based on config
    switchTool("draw");
    redrawCanvas(); // Draw initial placeholder
    console.log("Initialization complete.");
    console.log("Initial LAIBEL_CONFIG:", window.LAIBEL_CONFIG);
  }

  // Run initialization
  initializeApp();
}); // End DOMContentLoaded
