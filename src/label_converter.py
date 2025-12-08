"""
Label Format Converter
======================
Converts between different annotation formats: YOLO, COCO, Pascal VOC

Supported Formats:
- YOLO: class_id x_center y_center width height (normalized 0-1)
- COCO: JSON format with bounding boxes
- Pascal VOC: XML format

Educational Purpose:
Learn how different annotation formats work and how to convert between them.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import os


class LabelConverter:
    """
    Converts annotation files between different formats.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the converter.
        
        Args:
            class_names: List of class names (e.g., ['person', 'car', 'dog'])
        """
        self.class_names = class_names or []
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.id_to_class = {idx: name for idx, name in enumerate(self.class_names)}
    
    def parse_yolo_label(self, label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse YOLO format label file.
        
        YOLO Format (each line):
        class_id x_center y_center width height
        - All coordinates are normalized (0.0 to 1.0)
        - x_center, y_center: center of bounding box
        - width, height: box dimensions
        
        Args:
            label_path: Path to YOLO .txt file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of annotations as dictionaries
        """
        annotations = []
        
        if not os.path.exists(label_path):
            return annotations
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to absolute coordinates
                x_center_abs = x_center * img_width
                y_center_abs = y_center * img_height
                width_abs = width * img_width
                height_abs = height * img_height
                
                # Calculate corner coordinates
                x_min = x_center_abs - (width_abs / 2)
                y_min = y_center_abs - (height_abs / 2)
                x_max = x_center_abs + (width_abs / 2)
                y_max = y_center_abs + (height_abs / 2)
                
                annotations.append({
                    'class_id': class_id,
                    'class_name': self.id_to_class.get(class_id, f'class_{class_id}'),
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'x_center': x_center_abs,
                        'y_center': y_center_abs,
                        'width': width_abs,
                        'height': height_abs
                    },
                    'normalized': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    }
                })
        
        return annotations
    
    def write_yolo_label(self, annotations: List[Dict], output_path: str):
        """
        Write annotations to YOLO format file.
        
        Args:
            annotations: List of annotation dictionaries
            output_path: Path to save YOLO .txt file
        """
        with open(output_path, 'w') as f:
            for ann in annotations:
                norm = ann['normalized']
                line = f"{ann['class_id']} {norm['x_center']:.6f} {norm['y_center']:.6f} {norm['width']:.6f} {norm['height']:.6f}\n"
                f.write(line)
    
    def yolo_to_coco(self, image_dir: str, label_dir: str, output_path: str):
        """
        Convert YOLO format dataset to COCO JSON format.
        
        COCO Format:
        {
            "images": [...],
            "annotations": [...],
            "categories": [...]
        }
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO label files
            output_path: Path to save COCO JSON file
        """
        from PIL import Image
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for idx, class_name in enumerate(self.class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        image_id = 1
        
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        # Process each image
        for img_path in image_dir.glob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except:
                continue
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_width,
                "height": img_height
            })
            
            # Parse YOLO label
            label_path = label_dir / f"{img_path.stem}.txt"
            annotations = self.parse_yolo_label(str(label_path), img_width, img_height)
            
            # Add annotations
            for ann in annotations:
                bbox = ann['bbox']
                x_min = bbox['x_min']
                y_min = bbox['y_min']
                width = bbox['width']
                height = bbox['height']
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann['class_id'],
                    "bbox": [x_min, y_min, width, height],  # COCO format: [x, y, w, h]
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
            
            image_id += 1
        
        # Save COCO JSON
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ COCO format saved: {output_path}")
        print(f"   Images: {len(coco_data['images'])}")
        print(f"   Annotations: {len(coco_data['annotations'])}")
    
    def yolo_to_voc(self, image_path: str, label_path: str, output_dir: str):
        """
        Convert single YOLO label to Pascal VOC XML format.
        
        Pascal VOC Format:
        XML file with image metadata and bounding box annotations.
        
        Args:
            image_path: Path to image file
            label_path: Path to YOLO label file
            output_dir: Directory to save VOC XML file
        """
        from PIL import Image
        
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Parse YOLO label
        annotations = self.parse_yolo_label(label_path, img_width, img_height)
        
        # Create XML structure
        root = ET.Element("annotation")
        
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = Path(image_path).name
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"
        
        # Add objects
        for ann in annotations:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann['class_name']
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bbox = ann['bbox']
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox['x_min']))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox['y_min']))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox['x_max']))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox['y_max']))
        
        # Save XML
        tree = ET.ElementTree(root)
        output_path = Path(output_dir) / f"{Path(image_path).stem}.xml"
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def validate_yolo_labels(self, label_dir: str) -> Dict:
        """
        Validate YOLO label files for common issues.
        
        Args:
            label_dir: Directory containing YOLO label files
            
        Returns:
            Dictionary with validation results
        """
        issues = {
            'invalid_format': [],
            'out_of_bounds': [],
            'empty_files': [],
            'valid_files': 0
        }
        
        label_dir = Path(label_dir)
        
        for label_path in label_dir.glob('*.txt'):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines or all(not line.strip() for line in lines):
                issues['empty_files'].append(label_path.name)
                continue
            
            valid = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    issues['invalid_format'].append(f"{label_path.name}: {line}")
                    valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check if coordinates are normalized
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        issues['out_of_bounds'].append(f"{label_path.name}: {line}")
                        valid = False
                except:
                    issues['invalid_format'].append(f"{label_path.name}: {line}")
                    valid = False
            
            if valid:
                issues['valid_files'] += 1
        
        return issues


# Example usage
if __name__ == "__main__":
    # Initialize converter with class names
    converter = LabelConverter(class_names=['person', 'car', 'bicycle', 'dog'])
    
    # Validate labels
    print("🔍 Validating YOLO labels...")
    # issues = converter.validate_yolo_labels("path/to/labels")
    
    # Convert to COCO
    print("\n📦 Converting to COCO format...")
    # converter.yolo_to_coco("images_dir", "labels_dir", "output.json")
    
    print("\n✅ Label conversion complete!")
