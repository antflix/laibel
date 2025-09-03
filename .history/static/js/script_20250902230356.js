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

  // Dataset state
  let availableDatasets = [];
  let currentDataset = null;
  let currentSplit = null;

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
  const datasetUpload = document.getElementById("dataset-upload");
  const uploadDatasetBtn = document.getElementById("upload-dataset-btn");
  const datasetSelect = document.getElementById("dataset-select");
  const splitSelect = document.getElementById("split-select");
  const loadDatasetBtn = document.getElementById("load-dataset-btn");
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
  uploadDatasetBtn.addEventListener("click", () => {
    datasetUpload.click();
  });
  
  // Dataset management
  datasetUpload.addEventListener("change", handleDatasetUpload);
  datasetSelect.addEventListener("change", handleDatasetChange);
  splitSelect.addEventListener("change", handleSplitChange);
  loadDatasetBtn.addEventListener("click", loadSelectedDataset);

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

  // --- Dataset Management Functions ---
  async function loadAvailableDatasets() {
    try {
      console.log('Loading available datasets...');
      const response = await fetch('/api/datasets');
      const data = await response.json();
      
      if (data.success) {
        availableDatasets = data.datasets;
        console.log('Datasets loaded:', availableDatasets);
        updateDatasetSelect();
      } else {
        console.error('Failed to load datasets:', data.error);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  }

  function updateDatasetSelect() {
    datasetSelect.innerHTML = '<option value="">Select a dataset...</option>';
    
    availableDatasets.forEach(dataset => {
      const option = document.createElement('option');
      option.value = dataset.name;
      option.textContent = `${dataset.name} (${dataset.total_images} images, ${dataset.classes.length} classes)`;
      datasetSelect.appendChild(option);
    });
  }

  function updateSplitSelect() {
    splitSelect.innerHTML = '<option value="">All splits</option>';
    
    if (currentDataset && currentDataset.splits) {
      Object.keys(currentDataset.splits).forEach(splitName => {
        const splitInfo = currentDataset.splits[splitName];
        const option = document.createElement('option');
        option.value = splitName;
        option.textContent = `${splitName} (${splitInfo.image_count} images)`;
        splitSelect.appendChild(option);
      });
    }
  }

  async function handleDatasetUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.zip')) {
      alert('Please select a ZIP file containing your YOLO dataset.');
      return;
    }

    const formData = new FormData();
    formData.append('dataset', file);

    try {
      uploadDatasetBtn.disabled = true;
      uploadDatasetBtn.textContent = 'Uploading...';

      const response = await fetch('/upload_dataset', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.success) {
        alert(`Dataset uploaded successfully: ${result.message}`);
        await loadAvailableDatasets();
        
        datasetSelect.value = result.dataset.name;
        handleDatasetChange();
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed. Please check your internet connection and try again.');
    } finally {
      uploadDatasetBtn.disabled = false;
      uploadDatasetBtn.textContent = 'Upload Dataset';
      datasetUpload.value = '';
    }
  }

  function handleDatasetChange() {
    const selectedDatasetName = datasetSelect.value;
    currentDataset = availableDatasets.find(d => d.name === selectedDatasetName) || null;
    currentSplit = null;
    
    updateSplitSelect();
    updateNavigationUI();

    if (currentDataset && currentDataset.classes.length > 0) {
      labels = [];
      
      currentDataset.classes.forEach(className => {
        if (!labels.some(l => l.name === className)) {
          labels.push({
            name: className,
            color: getRandomColor()
          });
        }
      });
      
      updateLabelsList();
      updateNavigationUI();
    }
  }

  function handleSplitChange() {
    currentSplit = splitSelect.value || null;
    updateNavigationUI();
  }

  async function loadSelectedDataset() {
    if (!currentDataset) {
      alert('Please select a dataset first.');
      return;
    }

    try {
      loadDatasetBtn.disabled = true;
      loadDatasetBtn.textContent = 'Loading...';

      const url = `/api/datasets/${currentDataset.name}/images${currentSplit ? `?split=${currentSplit}` : ''}`;
      console.log('Fetching dataset images from:', url);
      
      const response = await fetch(url);
      const data = await response.json();

      if (data.success) {
        // Clear existing images
        imageData = [];
        currentImageIndex = -1;
        clearCanvasAndState();

        console.log(`Processing ${data.images.length} images from dataset`);

        // Process dataset images
        const imagePromises = data.images.map(async (imgInfo) => {
          try {
            console.log(`Loading image: ${imgInfo.name} from ${imgInfo.image_path}`);
            
            // Load image using the correct path
            const imageUrl = `/api/datasets/${currentDataset.name}/image/${encodeURIComponent(imgInfo.image_path)}`;
            console.log(`Fetching image from: ${imageUrl}`);
            
            const imageResponse = await fetch(imageUrl);
            if (!imageResponse.ok) {
              console.error(`Failed to fetch image ${imgInfo.name}: ${imageResponse.status} ${imageResponse.statusText}`);
              throw new Error('Failed to load image');
            }
            
            const imageBlob = await imageResponse.blob();
            console.log(`Image blob size: ${imageBlob.size} bytes, type: ${imageBlob.type}`);
            
            const imageObjectUrl = URL.createObjectURL(imageBlob);
            console.log(`Created object URL: ${imageObjectUrl}`);
            
            // Load labels
            const labelsResponse = await fetch(`/api/datasets/${currentDataset.name}/labels/${encodeURIComponent(imgInfo.name)}`);
            const labelsData = labelsResponse.ok ? await labelsResponse.json() : { boxes: [] };
            
            return new Promise((resolve, reject) => {
              const img = new Image();
              
              img.onload = function() {
                console.log(`Image loaded successfully: ${imgInfo.name}, size: ${img.width}x${img.height}`);
                
                let currentScaleRatio = 1;
                const currentOriginalWidth = img.width;
                const currentOriginalHeight = img.height;

                if (currentOriginalWidth > MAX_WIDTH || currentOriginalHeight > MAX_HEIGHT) {
                  const widthRatio = MAX_WIDTH / currentOriginalWidth;
                  const heightRatio = MAX_HEIGHT / currentOriginalHeight;
                  currentScaleRatio = Math.min(widthRatio, heightRatio);
                }

                console.log(`Scale ratio: ${currentScaleRatio}`);

                // Convert YOLO boxes to canvas coordinates
                const canvasBoxes = labelsData.boxes ? labelsData.boxes.map(box => ({
                  x: box.x * currentScaleRatio,
                  y: box.y * currentScaleRatio,
                  width: box.width * currentScaleRatio,
                  height: box.height * currentScaleRatio,
                  label: box.label
                })) : [];

                const data = {
                  src: imageObjectUrl, // Use the object URL
                  filename: imgInfo.name,
                  originalWidth: currentOriginalWidth,
                  originalHeight: currentOriginalHeight,
                  scaleRatio: currentScaleRatio,
                  boxes: canvasBoxes,
                  dataset: imgInfo.dataset,
                  split: imgInfo.split
                };
                
                console.log(`Image data prepared:`, data);
                resolve(data);
              };
              
              img.onerror = (error) => {
                console.error(`Failed to load image object for ${imgInfo.name}:`, error);
                console.error(`Image src was: ${imageObjectUrl}`);
                reject(new Error(`Failed to load image: ${imgInfo.name}`));
              };
              
              console.log(`Setting image src to: ${imageObjectUrl}`);
              img.src = imageObjectUrl;
            });
          } catch (error) {
            console.error(`Error processing image ${imgInfo.name}:`, error);
            return null;
          }
        });

        const results = await Promise.allSettled(imagePromises);
        const validImages = results
          .filter(result => result.status === 'fulfilled' && result.value)
          .map(result => result.value);

        console.log(`Processed ${validImages.length} valid images out of ${data.images.length} total`);
        imageData = validImages;

        if (imageData.length > 0) {
          console.log('Loading first image...');
          loadImageData(0);
          alert(`Loaded ${imageData.length} images from dataset "${currentDataset.name}"`);
        } else {
          alert('No valid images found in the selected dataset.');
        }

      } else {
        console.error('Dataset loading failed:', data.error);
        alert(`Failed to load dataset: ${data.error}`);
      }
    } catch (error) {
      console.error('Error loading dataset:', error);
      alert('Failed to load dataset. Please try again.');
    } finally {
      loadDatasetBtn.disabled = false;
      loadDatasetBtn.textContent = 'Load Dataset';
      updateNavigationUI();
    }
  }

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

  // --- Image Upload Handling (original functionality) ---
  imageUpload.addEventListener("change", (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    imageData = [];
    currentImageIndex = -1;
    clearCanvasAndState();

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

              if (currentOriginalWidth > MAX_WIDTH || currentOriginalHeight > MAX_HEIGHT) {
                const widthRatio = MAX_WIDTH / currentOriginalWidth;
                const heightRatio = MAX_HEIGHT / currentOriginalHeight;
                currentScaleRatio = Math.min(widthRatio, heightRatio);
              }

              const data = {
                src: event.target.result,
                filename: file.name,
                originalWidth: currentOriginalWidth,
                originalHeight: currentOriginalHeight,
                scaleRatio: currentScaleRatio,
                boxes: [],
              };
              imageData.push(data);
              resolve();
            };
            img.onerror = (err) => {
              console.error("Error loading image:", file.name, err);
              loadErrors++;
              reject(new Error(`Failed to load image: ${file.name}`));
            };
            img.src = event.target.result;
          };
          reader.onerror = (err) => {
            console.error("Error reading file:", file.name, err);
            loadErrors++;
            reject(new Error(`Failed to read file: ${file.name}`));
          };
          reader.readAsDataURL(file);
        });
        filePromises.push(loadPromise);
      }
    });

    Promise.allSettled(filePromises).then((results) => {
      const successfulLoads = results.filter(r => r.status === "fulfilled").length;
      console.log(`Processed ${files.length} files. Successfully loaded ${successfulLoads} images.`);

      if (imageData.length > 0) {
        loadImageData(0);
      } else {
        updateNavigationUI();
        if (loadErrors > 0) {
          alert(`Failed to load ${loadErrors} image(s). Please check console for details.`);
        }
      }
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

    console.log(`Loading image data for index ${index}:`, data);

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
      console.log(`Image onload fired for ${currentFilename}`);
      console.log(`Image natural dimensions: ${image.naturalWidth}x${image.naturalHeight}`);
      console.log(`Image current dimensions: ${image.width}x${image.height}`);
      
      const displayWidth = Math.round(originalWidth * scaleRatio);
      const displayHeight = Math.round(originalHeight * scaleRatio);

      console.log(`Setting canvas size to: ${displayWidth}x${displayHeight}`);

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
    
    image.onerror = (error) => {
      console.error("Error loading image source for display:", data.filename);
      console.error("Image src was:", data.src);
      console.error("Error details:", error);
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
    
    console.log(`Setting image src to: ${data.src}`);
    image.src = data.src; // Start loading the image from its blob URL
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
    const hasCurrentIndex = currentImageIndex >= 0 && currentImageIndex < imageData.length;
    const anyLoading = isYoloLoading || isYoloeLoading;
    const anyPredicting = isYoloPredicting || isYoloePredicting;
    const blockActions = anyLoading || anyPredicting;

    deleteImageBtn.disabled = !hasCurrentIndex || blockActions;
    prevImageBtn.disabled = currentImageIndex <= 0 || blockActions;
    nextImageBtn.disabled = currentImageIndex >= imageData.length - 1 || blockActions;
    drawBoxBtn.disabled = blockActions;
    editBoxBtn.disabled = blockActions;

    const hasDataset = currentDataset !== null;
    loadDatasetBtn.disabled = !hasDataset || blockActions;
    
    if (hasDataset && currentSplit) {
      loadDatasetBtn.textContent = `Load ${currentDataset.name} (${currentSplit})`;
    } else if (hasDataset) {
      loadDatasetBtn.textContent = `Load ${currentDataset.name}`;
    } else {
      loadDatasetBtn.textContent = 'Load Dataset';
    }

    if (!hasImages) {
      imageInfoSpan.textContent = "No images loaded";
    } else if (hasCurrentIndex) {
      const displayFilename = imageData[currentImageIndex].filename.length > 25
        ? imageData[currentImageIndex].filename.substring(0, 22) + "..."
        : imageData[currentImageIndex].filename;
      let infoText = `${currentImageIndex + 1} / ${imageData.length} (${displayFilename})`;
      
      const currentImg = imageData[currentImageIndex];
      if (currentImg.dataset && currentImg.split) {
        infoText += ` [${currentImg.split}]`;
      }
      
      imageInfoSpan.textContent = infoText;
    }

    // Model buttons update logic
    loadYoloModelBtn.disabled = isYoloModelLoaded || isYoloLoading;
    yoloAssistBtn.disabled = !hasCurrentIndex || !isYoloModelLoaded || isYoloPredicting || anyLoading;

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

    const currentLabelNames = labels.map(l => l.name);
    const requiredLabelsMatchLoaded = isYoloeModelLoaded &&
      yoloeModelClasses.length === currentLabelNames.length &&
      yoloeModelClasses.every(label => currentLabelNames.includes(label)) &&
      currentLabelNames.every(label => yoloeModelClasses.includes(label));

    loadYoloeModelBtn.disabled = isYoloeLoading || requiredLabelsMatchLoaded || labels.length === 0;
    yoloeAssistBtn.disabled = !hasCurrentIndex || !isYoloeModelLoaded || !requiredLabelsMatchLoaded || isYoloePredicting || anyLoading;

    if (isYoloeLoading) {
      loadYoloeModelBtn.textContent = "Loading YOLOE...";
    } else if (requiredLabelsMatchLoaded) {
      const displayClasses = yoloeModelClasses.length > 2
        ? yoloeModelClasses.slice(0, 2).join(", ") + "..."
        : yoloeModelClasses.join(", ") || "None";
      loadYoloeModelBtn.textContent = `YOLOE Loaded (${displayClasses})`;
    } else if (labels.length === 0) {
      loadYoloeModelBtn.textContent = "Add Labels to Load";
    } else {
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
  canvas.addEventListener("mouseleave", handleMouseLeave);

  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, canvas.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, canvas.height));
    return { x, y };
  }

  function getHandleUnderMouse(x, y) {
    if (!boxes) return null;
    for (let i = boxes.length - 1; i >= 0; i--) {
      const box = boxes[i];
      if (!box) continue;
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
    console.log('Redrawing canvas...');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const currentImageData =
      currentImageIndex >= 0 ? imageData[currentImageIndex] : null;

    if (image && canvas.width > 0 && canvas.height > 0) {
      console.log(`Drawing image on canvas: ${canvas.width}x${canvas.height}`);
      console.log(`Image ready state: width=${image.width}, height=${image.height}, complete=${image.complete}`);
      
      try {
        // Make sure image is loaded before drawing
        if (image.complete && image.naturalHeight !== 0) {
          ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
          console.log('Image drawn successfully');
        } else {
          console.warn('Image not ready for drawing yet');
          drawPlaceholder("Loading image...");
          return;
        }
      } catch (e) {
        console.error("Error drawing image:", e);
        drawPlaceholder("Error drawing image");
        return;
      }
    } else {
      console.log('No image to draw or canvas size is zero');
      drawPlaceholder("Upload images or load dataset to begin");
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

    console.log(`Drawing ${currentBoxes.length} boxes`);
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
    console.log(`Drawing placeholder: ${text}`);
    ctx.fillStyle = "#33373e";
    ctx.fillRect(0, 0, canvas.width || 640, canvas.height || 480);
    ctx.fillStyle = "#828a9a";
    ctx.font = "16px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, (canvas.width || 640) / 2, (canvas.height || 480) / 2);
    ctx.textAlign = "start"; // Reset text alignment
    ctx.textBaseline = "alphabetic"; // Reset text baseline
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

  // --- Stub functions for model functionality ---
  async function handleLoadYoloModel() {
    // Placeholder for YOLO model loading
    console.log('YOLO model loading requested');
  }

  async function handleLoadYoloeModel() {
    // Placeholder for YOLOE model loading
    console.log('YOLOE model loading requested');
  }

  async function handleYoloAssist() {
    // Placeholder for YOLO assist
    console.log('YOLO assist requested');
  }

  async function handleYoloeAssist() {
    // Placeholder for YOLOE assist
    console.log('YOLOE assist requested');
  }

  function saveJsonAnnotations() {
    // Placeholder for JSON export
    console.log('JSON export requested');
  }

  function exportYoloAnnotations() {
    // Placeholder for YOLO export
    console.log('YOLO export requested');
  }

  function deleteCurrentImage() {
    // Placeholder for image deletion
    console.log('Image deletion requested');
  }

  function handleKeyDown(event) {
    // Placeholder for keyboard shortcuts
    console.log('Key pressed:', event.key);
  }

  // --- Initial Setup on Page Load ---
  function initializeApp() {
    console.log("Initializing Laibel Application...");
    updateLabelsList();
    updateAnnotationsList();
    initializeModelButtons(); // Sets up initial button states based on config
    loadAvailableDatasets(); // Load available datasets
    switchTool("draw");
    redrawCanvas(); // Draw initial placeholder
    console.log("Initialization complete.");
    console.log("Initial LAIBEL_CONFIG:", window.LAIBEL_CONFIG);
  }
