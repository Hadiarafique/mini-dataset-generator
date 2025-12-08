"""
ZIP Creator Module
==================
This module creates downloadable ZIP archives of YOLO datasets.

What it does:
- Takes a dataset folder
- Compresses it into a .zip file
- Makes it easy to share or download
- Preserves folder structure
"""

import zipfile
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class ZipCreator:
    """
    Creates ZIP archives of datasets for easy sharing and downloading.
    """
    
    def __init__(self):
        """Initialize the ZIP creator."""
        pass
    
    def create_zip(self, 
                   source_folder: str, 
                   output_path: Optional[str] = None,
                   zip_name: Optional[str] = None) -> str:
        """
        Create a ZIP file from a folder.
        
        Args:
            source_folder: Path to the folder to compress
            output_path: Where to save the ZIP file (default: same as source folder parent)
            zip_name: Name of the ZIP file (default: folder_name_timestamp.zip)
            
        Returns:
            Path to the created ZIP file
        """
        source_path = Path(source_folder)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")
        
        # Generate ZIP filename if not provided
        if zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"{source_path.name}_{timestamp}.zip"
        
        # Make sure zip_name ends with .zip
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'
        
        # Determine output path
        if output_path is None:
            output_path = source_path.parent
        else:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Full path to ZIP file
        zip_file_path = output_path / zip_name
        
        print(f"\n📦 Creating ZIP archive...")
        print(f"   Source: {source_path}")
        print(f"   Output: {zip_file_path}")
        
        # Create the ZIP file
        file_count = 0
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in source folder
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    # Get full file path
                    file_path = Path(root) / file
                    
                    # Get relative path (for maintaining structure in ZIP)
                    arcname = file_path.relative_to(source_path.parent)
                    
                    # Add to ZIP
                    zipf.write(file_path, arcname)
                    file_count += 1
                    
                    if file_count % 10 == 0:
                        print(f"   Compressed {file_count} files...")
        
        # Get file size
        zip_size_mb = zip_file_path.stat().st_size / (1024 * 1024)
        
        print(f"\n✅ ZIP creation complete!")
        print(f"   📁 Files compressed: {file_count}")
        print(f"   💾 ZIP size: {zip_size_mb:.2f} MB")
        print(f"   📍 Location: {zip_file_path}")
        
        return str(zip_file_path)
    
    def create_dataset_zip(self, 
                          dataset_path: str,
                          include_augmented: bool = False,
                          augmented_path: Optional[str] = None) -> str:
        """
        Create a ZIP specifically for YOLO datasets.
        Optionally includes augmented images folder.
        
        Args:
            dataset_path: Path to the YOLO dataset folder
            include_augmented: Whether to include augmented images folder
            augmented_path: Path to augmented images (if include_augmented=True)
            
        Returns:
            Path to the created ZIP file
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Generate ZIP name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{dataset_path.name}_yolo_dataset_{timestamp}.zip"
        
        # Output in same directory as dataset
        output_path = dataset_path.parent
        zip_file_path = output_path / zip_name
        
        print(f"\n📦 Creating YOLO Dataset ZIP...")
        print(f"   Dataset: {dataset_path.name}")
        
        file_count = 0
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add dataset folder
            print(f"   📁 Adding dataset files...")
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(dataset_path.parent)
                    zipf.write(file_path, arcname)
                    file_count += 1
            
            # Optionally add augmented images
            if include_augmented and augmented_path:
                aug_path = Path(augmented_path)
                if aug_path.exists():
                    print(f"   🎨 Adding augmented images...")
                    for root, dirs, files in os.walk(aug_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(aug_path.parent)
                            zipf.write(file_path, arcname)
                            file_count += 1
        
        # Get file size
        zip_size_mb = zip_file_path.stat().st_size / (1024 * 1024)
        
        print(f"\n✅ Dataset ZIP ready for download!")
        print(f"   📁 Total files: {file_count}")
        print(f"   💾 Size: {zip_size_mb:.2f} MB")
        print(f"   📍 Location: {zip_file_path}")
        
        return str(zip_file_path)
    
    def get_zip_info(self, zip_path: str) -> dict:
        """
        Get information about a ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Dictionary with ZIP file information
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Calculate uncompressed size
            uncompressed_size = sum(info.file_size for info in zipf.infolist())
            compressed_size = zip_path.stat().st_size
            
            compression_ratio = (1 - compressed_size / uncompressed_size) * 100
            
            info = {
                'path': str(zip_path),
                'name': zip_path.name,
                'file_count': len(file_list),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'files': file_list
            }
            
            return info
    
    def extract_zip(self, zip_path: str, extract_to: Optional[str] = None) -> str:
        """
        Extract a ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            extract_to: Where to extract (default: same directory as ZIP)
            
        Returns:
            Path to extraction directory
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        # Determine extraction path
        if extract_to is None:
            extract_to = zip_path.parent / zip_path.stem
        else:
            extract_to = Path(extract_to)
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 Extracting ZIP...")
        print(f"   From: {zip_path}")
        print(f"   To: {extract_to}")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
            file_count = len(zipf.namelist())
        
        print(f"✅ Extraction complete! ({file_count} files)")
        
        return str(extract_to)


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs only when you execute this file directly
    
    creator = ZipCreator()
    
    # Get parent directory
    current_dir = Path(__file__).parent.parent
    output_folder = current_dir / "output"
    
    # Check if there's a dataset to zip
    if output_folder.exists():
        # Find first dataset folder
        dataset_folders = [f for f in output_folder.iterdir() if f.is_dir() and f.name != 'augmented_images']
        
        if dataset_folders:
            test_dataset = dataset_folders[0]
            print(f"🧪 Testing ZIP creation with: {test_dataset.name}")
            
            # Create ZIP
            zip_file = creator.create_zip(
                source_folder=str(test_dataset),
                output_path=str(output_folder)
            )
            
            # Show ZIP info
            print(f"\n📊 ZIP Information:")
            info = creator.get_zip_info(zip_file)
            print(f"   Files: {info['file_count']}")
            print(f"   Size: {info['compressed_size_mb']:.2f} MB")
            print(f"   Compression: {info['compression_ratio']:.1f}%")
            
            print(f"\n✅ Test complete!")
        else:
            print("⚠️  No datasets found in output folder")
            print("   Run main.py first to create a dataset")
    else:
        print(f"⚠️  Output folder not found: {output_folder}")
        print("   Run main.py first to create a dataset")
