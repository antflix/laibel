// Complete script.js file - Part 2
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

  // Run initialization
  initializeApp();
}); // End DOMContentLoaded