#!/usr/bin/env python3
"""
Dataset Debug Tool for Laibel
Run this script to debug your dataset structure and see what the app detects
"""

import os
import sys
from pathlib import Path
import yaml

def check_dataset_structure(dataset_path):
    """Debug function to check dataset structure"""
    dataset_path = Path(dataset_path)
    
    print(f"=== Analyzing Dataset: {dataset_path.name} ===")
    print(f"Full path: {dataset_path}")
    print(f"Exists: {dataset_path.exists()}")
    print(f"Is directory: {dataset_path.is_dir()}")
    
    if not dataset_path.exists():
        print("❌ Dataset path doesn't exist!")
        return False
    
    # List all contents
    print(f"\n📂 Directory contents:")
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    # Check for YAML files
    yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
    print(f"\n🔧 YAML files found: {len(yaml_files)}")
    for yaml_file in yaml_files:
        print(f"  📄 {yaml_file.name}")
        try:
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            print(f"    ✅ Valid YAML with keys: {list(yaml_data.keys())}")
            if 'names' in yaml_data:
                print(f"    📋 Classes: {yaml_data['names']}")
        except Exception as e:
            print(f"    ❌ Error reading YAML: {e}")
    
    # Check for standard YOLO structure
    splits = ['train', 'val', 'test', 'valid']
    total_images = 0
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            print(f"\n📂 Split '{split}' found:")
            
            # Check for images and labels directories
            images_dir = split_path / 'images' if (split_path / 'images').exists() else split_path
            labels_dir = split_path / 'labels' if (split_path / 'labels').exists() else None
            
            print(f"  📁 Images directory: {images_dir}")
            print(f"  📁 Labels directory: {labels_dir}")
            
            # Count files
            image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
            images = []
            
            for ext in image_extensions:
                images.extend(list(images_dir.glob(f'*.{ext}')))
                images.extend(list(images_dir.glob(f'*.{ext.upper()}')))
            
            labels = []
            if labels_dir and labels_dir.exists():
                labels = list(labels_dir.glob('*.txt'))
            
            print(f"  🖼️  Images found: {len(images)}")
            print(f"  🏷️  Labels found: {len(labels)}")
            
            if images:
                print(f"  📝 Sample images: {[img.name for img in images[:3]]}")
                total_images += len(images)
            
            if labels:
                print(f"  📝 Sample labels: {[lbl.name for lbl in labels[:3]]}")
                
                # Check first label file
                try:
                    with open(labels[0], 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            parts = first_line.split()
                            print(f"  🔍 Sample label line: {first_line}")
                            print(f"  🔍 Parts count: {len(parts)} (should be 5)")
                except Exception as e:
                    print(f"  ❌ Error reading label: {e}")
        else:
            print(f"📂 Split '{split}' not found")
    
    print(f"\n📊 Summary:")
    print(f"  Total images across all splits: {total_images}")
    print(f"  Valid dataset: {'✅ YES' if total_images > 0 else '❌ NO'}")
    
    return total_images > 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_dataset.py <path_to_dataset>")
        print("Example: python debug_dataset.py static/uploads/my_dataset")
        return
    
    dataset_path = sys.argv[1]
    
    # If it's in uploads folder, scan all datasets
    if dataset_path.endswith('uploads') or dataset_path == 'static/uploads':
        uploads_path = Path(dataset_path)
        if uploads_path.exists():
            print(f"🔍 Scanning all datasets in: {uploads_path}")
            datasets_found = 0
            for item in uploads_path.iterdir():
                if item.is_dir():
                    print(f"\n{'='*50}")
                    is_valid = check_dataset_structure(item)
                    if is_valid:
                        datasets_found += 1
            print(f"\n🎯 Total valid datasets found: {datasets_found}")
        else:
            print(f"❌ Uploads directory not found: {uploads_path}")
    else:
        # Check single dataset
        check_dataset_structure(dataset_path)

if __name__ == "__main__":
    main()