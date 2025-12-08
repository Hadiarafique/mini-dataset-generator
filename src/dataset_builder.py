"""
YOLO Dataset Builder
====================
This module creates the proper folder structure for YOLO training.

YOLO Dataset Structure:
dataset/
  ├── train/
  │   ├── images/     <- Training images
  │   └── labels/     <- Training labels (.txt files)
  ├── test/
  │   ├── images/     <- Test images
  │   └── labels/     <- Test labels
  ├── valid/
  │   ├── images/     <- Validation images
  │   └── labels/     <- Validation labels
  └── data.yaml       <- Configuration file

What this module does:
1. Creates the folder structure
2. Splits images into train/test/valid sets (e.g., 70% train, 15% test, 15% validation)
3. Generates the data.yaml file
4. Optionally creates empty label files (for later annotation)
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import random


class YOLODatasetBuilder:
    """
    Creates YOLO-format dataset structure from a list of images.
    """
    
    def __init__(self, dataset_name: str = "my_dataset"):
        """
        Initialize the dataset builder.
        
        Args:
            dataset_name: Name of the dataset (used for folder name)
        """
        self.dataset_name = dataset_name
        self.train_ratio = 0.7   # 70% for training
        self.test_ratio = 0.15   # 15% for testing
        self.valid_ratio = 0.15  # 15% for validation
    
    def create_folder_structure(self, output_path: str) -> Dict[str, Path]:
        """
        Create the YOLO folder structure.
        
        Args:
            output_path: Base path where dataset will be created
            
        Returns:
            Dictionary with paths to all created folders
        """
        base_path = Path(output_path) / self.dataset_name
        
        # Define all folders we need
        folders = {
            'base': base_path,
            'train_images': base_path / 'train' / 'images',
            'train_labels': base_path / 'train' / 'labels',
            'test_images': base_path / 'test' / 'images',
            'test_labels': base_path / 'test' / 'labels',
            'valid_images': base_path / 'valid' / 'images',
            'valid_labels': base_path / 'valid' / 'labels',
        }
        
        # Create all folders
        print(f"\n📁 Creating YOLO dataset structure: {self.dataset_name}")
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {folder_name}: {folder_path}")
        
        return folders
    
    def split_train_test_valid(self, image_paths: List[str], train_ratio: float = None, 
                               test_ratio: float = None, valid_ratio: float = None) -> Tuple[List[str], List[str], List[str]]:
        """
        Split images into training, test, and validation sets.
        
        Args:
            image_paths: List of image file paths
            train_ratio: Ratio of training images (default: 0.7 = 70%)
            test_ratio: Ratio of test images (default: 0.15 = 15%)
            valid_ratio: Ratio of validation images (default: 0.15 = 15%)
            
        Returns:
            Tuple of (train_images, test_images, valid_images)
        """
        if train_ratio is not None:
            self.train_ratio = train_ratio
        if test_ratio is not None:
            self.test_ratio = test_ratio
        if valid_ratio is not None:
            self.valid_ratio = valid_ratio
        
        # Validate ratios sum to 1.0
        total = self.train_ratio + self.test_ratio + self.valid_ratio
        if abs(total - 1.0) > 0.01:
            print(f"⚠️  Warning: Ratios sum to {total:.2f}, normalizing to 1.0")
            self.train_ratio /= total
            self.test_ratio /= total
            self.valid_ratio /= total
        
        # Shuffle images randomly
        images_copy = image_paths.copy()
        random.shuffle(images_copy)
        
        # Calculate split points
        total_images = len(images_copy)
        train_end = int(total_images * self.train_ratio)
        test_end = train_end + int(total_images * self.test_ratio)
        
        train_images = images_copy[:train_end]
        test_images = images_copy[train_end:test_end]
        valid_images = images_copy[test_end:]
        
        print(f"\n✂️  Splitting dataset:")
        print(f"  📊 Total images: {len(image_paths)}")
        print(f"  🎓 Training: {len(train_images)} ({self.train_ratio*100:.1f}%)")
        print(f"  🧪 Test: {len(test_images)} ({self.test_ratio*100:.1f}%)")
        print(f"  ✅ Validation: {len(valid_images)} ({self.valid_ratio*100:.1f}%)")
        
        return train_images, test_images, valid_images
    
    def copy_images_to_dataset(self, image_paths: List[str], destination_folder: Path) -> List[str]:
        """
        Copy images to the destination folder.
        
        Args:
            image_paths: List of source image paths
            destination_folder: Where to copy images
            
        Returns:
            List of destination file paths
        """
        copied_files = []
        
        for img_path in image_paths:
            src = Path(img_path)
            dst = destination_folder / src.name
            
            try:
                shutil.copy2(src, dst)
                copied_files.append(str(dst))
            except Exception as e:
                print(f"  ❌ Error copying {src.name}: {e}")
        
        return copied_files
    
    def create_empty_labels(self, image_paths: List[str], labels_folder: Path) -> List[str]:
        """
        Create empty label files for each image.
        (Users can annotate these later using tools like LabelImg or Roboflow)
        
        Args:
            image_paths: List of image file paths
            labels_folder: Folder to save label files
            
        Returns:
            List of created label file paths
        """
        label_files = []
        
        for img_path in image_paths:
            # Get image filename without extension
            img_name = Path(img_path).stem
            
            # Create corresponding .txt label file
            label_path = labels_folder / f"{img_name}.txt"
            
            # Create empty file (will be filled during annotation)
            label_path.touch()
            label_files.append(str(label_path))
        
        return label_files
    
    def generate_yaml(self, dataset_path: Path, class_names: List[str]) -> str:
        """
        Generate the data.yaml configuration file for YOLO.
        
        Args:
            dataset_path: Path to the dataset folder
            class_names: List of object class names (e.g., ['apple', 'banana'])
            
        Returns:
            Path to the generated YAML file
        """
        # Create YAML configuration
        config = {
            'path': str(dataset_path.absolute()),  # Dataset root directory
            'train': 'train/images',  # Path to training images (relative to 'path')
            'test': 'test/images',    # Path to test images
            'val': 'valid/images',    # Path to validation images
            
            'names': {i: name for i, name in enumerate(class_names)},  # Class names
            'nc': len(class_names),  # Number of classes
        }
        
        # Write YAML file
        yaml_path = dataset_path / 'data.yaml'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n📄 Generated data.yaml:")
        print(f"  Path: {yaml_path}")
        print(f"  Classes: {class_names}")
        print(f"  Number of classes: {len(class_names)}")
        
        return str(yaml_path)
    
    def build_dataset(self, 
                     image_paths: List[str], 
                     output_path: str, 
                     class_names: List[str] = None,
                     train_ratio: float = 0.7,
                     test_ratio: float = 0.15,
                     valid_ratio: float = 0.15,
                     create_labels: bool = True) -> Dict[str, any]:
        """
        Complete workflow: Build entire YOLO dataset from images.
        
        Args:
            image_paths: List of source image paths
            output_path: Where to create the dataset
            class_names: List of object classes (default: ['object'])
            train_ratio: Training split ratio (default: 0.7 = 70%)
            test_ratio: Test split ratio (default: 0.15 = 15%)
            valid_ratio: Validation split ratio (default: 0.15 = 15%)
            create_labels: Whether to create empty label files
            
        Returns:
            Dictionary with dataset information
        """
        if class_names is None:
            class_names = ['object']  # Default class name
        
        print(f"\n🚀 Building YOLO dataset: {self.dataset_name}")
        print(f"   Input: {len(image_paths)} images")
        print(f"   Split: {train_ratio*100:.0f}% train, {test_ratio*100:.0f}% test, {valid_ratio*100:.0f}% valid")
        
        # Step 1: Create folder structure
        folders = self.create_folder_structure(output_path)
        
        # Step 2: Split images
        train_images, test_images, valid_images = self.split_train_test_valid(image_paths, train_ratio, test_ratio, valid_ratio)
        
        # Step 3: Copy images to train folder
        print(f"\n📋 Copying training images...")
        train_copied = self.copy_images_to_dataset(train_images, folders['train_images'])
        print(f"  ✅ Copied {len(train_copied)} training images")
        
        # Step 4: Copy images to test folder
        print(f"\n📋 Copying test images...")
        test_copied = self.copy_images_to_dataset(test_images, folders['test_images'])
        print(f"  ✅ Copied {len(test_copied)} test images")
        
        # Step 5: Copy images to valid folder
        print(f"\n📋 Copying validation images...")
        valid_copied = self.copy_images_to_dataset(valid_images, folders['valid_images'])
        print(f"  ✅ Copied {len(valid_copied)} validation images")
        
        # Step 6: Create empty label files (if requested)
        if create_labels:
            print(f"\n🏷️  Creating empty label files...")
            train_labels = self.create_empty_labels(train_copied, folders['train_labels'])
            test_labels = self.create_empty_labels(test_copied, folders['test_labels'])
            valid_labels = self.create_empty_labels(valid_copied, folders['valid_labels'])
            print(f"  ✅ Created {len(train_labels)} train labels")
            print(f"  ✅ Created {len(test_labels)} test labels")
            print(f"  ✅ Created {len(valid_labels)} valid labels")
        
        # Step 7: Generate YAML configuration
        yaml_path = self.generate_yaml(folders['base'], class_names)
        
        # Summary
        result = {
            'dataset_path': str(folders['base']),
            'yaml_path': yaml_path,
            'train_images': len(train_copied),
            'test_images': len(test_copied),
            'valid_images': len(valid_copied),
            'total_images': len(train_copied) + len(test_copied) + len(valid_copied),
            'classes': class_names,
        }
        
        print(f"\n" + "="*60)
        print(f"✅ DATASET BUILD COMPLETE!")
        print(f"="*60)
        print(f"📦 Dataset: {folders['base']}")
        print(f"📄 Config: {yaml_path}")
        print(f"🎓 Training: {len(train_copied)} images")
        print(f"🧪 Test: {len(test_copied)} images")
        print(f"✅ Validation: {len(valid_copied)} images")
        print(f"🏷️  Classes: {', '.join(class_names)}")
        print(f"="*60)
        
        return result


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs only when you execute this file directly
    
    builder = YOLODatasetBuilder(dataset_name="test_dataset")
    
    # Get parent directory
    current_dir = Path(__file__).parent.parent
    test_folder = current_dir / "test_images"
    output_folder = current_dir / "output"
    
    if test_folder.exists():
        # Get all images from test folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend([str(f) for f in test_folder.glob(ext)])
        
        if image_files:
            print(f"🧪 Testing dataset builder with {len(image_files)} images")
            
            # Build dataset
            result = builder.build_dataset(
                image_paths=image_files,
                output_path=str(output_folder),
                class_names=['apple', 'orange'],  # Example classes
                train_ratio=0.8,
                create_labels=True
            )
            
            print(f"\n✅ Test complete!")
            print(f"   Check: {result['dataset_path']}")
        else:
            print("⚠️  No images found in test_images folder")
    else:
        print(f"⚠️  Test folder not found: {test_folder}")
        print("Please add some images to the 'test_images' folder first!")
