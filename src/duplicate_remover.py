"""
Duplicate Image Remover
=======================
This module removes duplicate images using perceptual hashing.

How it works:
1. Perceptual hash converts each image into a "fingerprint" (hash)
2. Similar images have similar hashes
3. We compare hashes to find duplicates
4. Keep only unique images

Key Concept: Perceptual hashing is different from regular hashing.
Regular hash: Even 1 pixel change = completely different hash
Perceptual hash: Similar images = similar hash (even with small changes)
"""

import os
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict


class DuplicateRemover:
    """
    Removes duplicate images from a folder using perceptual hashing.
    """
    
    def __init__(self, threshold: int = 5):
        """
        Initialize the duplicate remover.
        
        Args:
            threshold: How similar images need to be to count as duplicates.
                      Lower = more strict (0 = exact match only)
                      Higher = more lenient (10 = very similar images)
                      Default 5 is a good balance
        """
        self.threshold = threshold
        self.image_hashes: Dict[str, str] = {}  # Store hash for each image
        
    def calculate_hash(self, image_path: str) -> str:
        """
        Calculate perceptual hash for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Hexadecimal hash string
        """
        try:
            # Open image and convert to hash
            img = Image.open(image_path)
            # phash = perceptual hash (good for finding similar images)
            hash_value = imagehash.phash(img)
            return str(hash_value)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def is_duplicate(self, hash1: str, hash2: str) -> bool:
        """
        Check if two hashes represent duplicate images.
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            
        Returns:
            True if images are duplicates, False otherwise
        """
        # Calculate Hamming distance (number of different bits)
        hash1_obj = imagehash.hex_to_hash(hash1)
        hash2_obj = imagehash.hex_to_hash(hash2)
        distance = hash1_obj - hash2_obj
        
        # If distance is less than threshold, they're duplicates
        return distance <= self.threshold
    
    def find_duplicates(self, image_folder: str) -> Tuple[List[str], List[str]]:
        """
        Find and separate unique images from duplicates.
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            Tuple of (unique_images, duplicate_images)
        """
        image_folder = Path(image_folder)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Get all image files
        image_files = [
            f for f in image_folder.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        unique_images = []
        duplicate_images = []
        unique_hashes = []
        
        print(f"\n🔍 Scanning {len(image_files)} images for duplicates...")
        
        for img_file in image_files:
            # Calculate hash for current image
            current_hash = self.calculate_hash(str(img_file))
            
            if current_hash is None:
                continue
            
            # Check if this image is a duplicate of any unique image
            is_dup = False
            for unique_hash in unique_hashes:
                if self.is_duplicate(current_hash, unique_hash):
                    duplicate_images.append(str(img_file))
                    is_dup = True
                    print(f"  ❌ Duplicate found: {img_file.name}")
                    break
            
            # If not a duplicate, add to unique list
            if not is_dup:
                unique_images.append(str(img_file))
                unique_hashes.append(current_hash)
                print(f"  ✅ Unique: {img_file.name}")
        
        print(f"\n📊 Results:")
        print(f"  Unique images: {len(unique_images)}")
        print(f"  Duplicates found: {len(duplicate_images)}")
        
        return unique_images, duplicate_images
    
    def remove_duplicates(self, image_folder: str, delete: bool = False) -> List[str]:
        """
        Remove duplicates from a folder.
        
        Args:
            image_folder: Path to folder containing images
            delete: If True, actually delete duplicate files
                   If False, just return list of unique images
        
        Returns:
            List of unique image paths
        """
        unique_images, duplicate_images = self.find_duplicates(image_folder)
        
        if delete and duplicate_images:
            print(f"\n🗑️  Deleting {len(duplicate_images)} duplicate files...")
            for dup in duplicate_images:
                try:
                    os.remove(dup)
                    print(f"  Deleted: {Path(dup).name}")
                except Exception as e:
                    print(f"  Error deleting {dup}: {e}")
        
        return unique_images


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs only when you execute this file directly
    
    # Example: Remove duplicates from test_images folder
    remover = DuplicateRemover(threshold=5)
    
    # Get parent directory (mini_dataset_generator)
    current_dir = Path(__file__).parent.parent
    test_folder = current_dir / "test_images"
    
    if test_folder.exists():
        unique = remover.remove_duplicates(str(test_folder), delete=False)
        print(f"\n✅ Done! Found {len(unique)} unique images")
    else:
        print(f"⚠️  Test folder not found: {test_folder}")
        print("Please add some images to the 'test_images' folder first!")
