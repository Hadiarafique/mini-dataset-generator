"""
Image Augmentation Module
==========================
This module applies various transformations to images to increase dataset size.

What is Augmentation?
- Taking 1 image and creating multiple versions of it
- Each version has slight changes (rotation, flip, brightness, etc.)
- Helps ML models learn better by seeing more variations

Example: 10 original images → 100 augmented images
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Dict, Tuple
import random


class ImageAugmenter:
    """
    Applies augmentation transforms to images to increase dataset size.
    """
    
    def __init__(self):
        """
        Initialize the augmenter with predefined transformations.
        """
        # Define augmentation pipeline using albumentations
        # Each transform has a probability (p) of being applied
        self.transform = A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),              # Flip left-right
            A.VerticalFlip(p=0.3),                # Flip top-bottom
            A.Rotate(limit=45, p=0.7),            # Rotate up to 45 degrees
            A.ShiftScaleRotate(                   # Combined transform
                shift_limit=0.1,                   # Shift up to 10%
                scale_limit=0.2,                   # Scale up to 20%
                rotate_limit=30,                   # Rotate up to 30 degrees
                p=0.5
            ),
            
            # Color/brightness transforms
            A.RandomBrightnessContrast(           # Change brightness/contrast
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(                 # Change colors
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4
            ),
            
            # Blur and noise
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Slight blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add noise
            
            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), p=0.3),  # Perspective change
        ])
        
        # Simple transforms (without albumentations) for learning
        self.simple_transforms = {
            'rotate_90': self._rotate_90,
            'rotate_180': self._rotate_180,
            'rotate_270': self._rotate_270,
            'flip_horizontal': self._flip_horizontal,
            'flip_vertical': self._flip_vertical,
            'brightness_up': self._brightness_up,
            'brightness_down': self._brightness_down,
        }
    
    # Simple transformation methods (easier to understand)
    
    def _rotate_90(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 90 degrees clockwise"""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def _rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees"""
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def _rotate_270(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 270 degrees clockwise (90 counter-clockwise)"""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally (left-right)"""
        return cv2.flip(image, 1)
    
    def _flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically (top-bottom)"""
        return cv2.flip(image, 0)
    
    def _brightness_up(self, image: np.ndarray) -> np.ndarray:
        """Increase brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _brightness_down(self, image: np.ndarray) -> np.ndarray:
        """Decrease brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def augment_image(self, image_path: str, num_augmentations: int = 5) -> List[Tuple[np.ndarray, str]]:
        """
        Generate augmented versions of a single image.
        
        Args:
            image_path: Path to the input image
            num_augmentations: How many augmented versions to create
            
        Returns:
            List of (augmented_image, transform_name) tuples
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return []
        
        augmented_images = []
        image_name = Path(image_path).stem  # Get filename without extension
        
        print(f"\n🎨 Augmenting: {Path(image_path).name}")
        
        # Generate augmented versions
        for i in range(num_augmentations):
            # Use albumentations for random transforms
            transformed = self.transform(image=image)
            aug_image = transformed['image']
            
            transform_name = f"{image_name}_aug_{i+1}"
            augmented_images.append((aug_image, transform_name))
            print(f"  ✅ Created: {transform_name}")
        
        return augmented_images
    
    def augment_simple(self, image_path: str, transforms: List[str] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Apply specific simple transformations (for learning/testing).
        
        Args:
            image_path: Path to the input image
            transforms: List of transform names to apply (e.g., ['rotate_90', 'flip_horizontal'])
                       If None, applies all simple transforms
            
        Returns:
            List of (augmented_image, transform_name) tuples
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return []
        
        augmented_images = []
        image_name = Path(image_path).stem
        
        # If no transforms specified, use all
        if transforms is None:
            transforms = list(self.simple_transforms.keys())
        
        print(f"\n🎨 Applying simple transforms to: {Path(image_path).name}")
        
        for transform_name in transforms:
            if transform_name in self.simple_transforms:
                # Apply the transformation
                transform_func = self.simple_transforms[transform_name]
                aug_image = transform_func(image.copy())
                
                output_name = f"{image_name}_{transform_name}"
                augmented_images.append((aug_image, output_name))
                print(f"  ✅ Applied: {transform_name}")
            else:
                print(f"  ⚠️  Unknown transform: {transform_name}")
        
        return augmented_images
    
    def augment_batch(self, image_paths: List[str], num_augmentations: int = 5) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        """
        Augment multiple images at once.
        
        Args:
            image_paths: List of image file paths
            num_augmentations: Number of augmented versions per image
            
        Returns:
            Dictionary mapping original image path to list of augmented images
        """
        results = {}
        
        print(f"\n🚀 Starting batch augmentation for {len(image_paths)} images...")
        print(f"   Creating {num_augmentations} augmentations per image")
        print(f"   Total output: {len(image_paths) * num_augmentations} new images\n")
        
        for img_path in image_paths:
            augmented = self.augment_image(img_path, num_augmentations)
            results[img_path] = augmented
        
        total_generated = sum(len(augs) for augs in results.values())
        print(f"\n✅ Batch complete! Generated {total_generated} augmented images")
        
        return results
    
    def save_augmented_images(self, augmented_images: List[Tuple[np.ndarray, str]], output_folder: str) -> List[str]:
        """
        Save augmented images to disk.
        
        Args:
            augmented_images: List of (image, name) tuples
            output_folder: Folder to save images
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for image, name in augmented_images:
            file_path = output_path / f"{name}.jpg"
            cv2.imwrite(str(file_path), image)
            saved_paths.append(str(file_path))
        
        print(f"\n💾 Saved {len(saved_paths)} images to: {output_folder}")
        
        return saved_paths


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs only when you execute this file directly
    
    augmenter = ImageAugmenter()
    
    # Get parent directory
    current_dir = Path(__file__).parent.parent
    test_folder = current_dir / "test_images"
    output_folder = current_dir / "output" / "augmented_test"
    
    if test_folder.exists():
        # Get first image from test folder
        image_files = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.png"))
        
        if image_files:
            test_image = str(image_files[0])
            print(f"🧪 Testing augmentation on: {Path(test_image).name}")
            
            # Test simple transforms
            augmented = augmenter.augment_simple(test_image)
            
            if augmented:
                # Save results
                augmenter.save_augmented_images(augmented, str(output_folder))
                print(f"\n✅ Test complete! Check the 'output/augmented_test' folder")
        else:
            print("⚠️  No images found in test_images folder")
    else:
        print(f"⚠️  Test folder not found: {test_folder}")
        print("Please add some images to the 'test_images' folder first!")
