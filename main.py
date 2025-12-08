"""
Main Script - Phase 1 & 2 Complete Pipeline
============================================
This is the main script that combines all Phase 1 & 2 modules:
1. Remove duplicates
2. Augment images
3. Build YOLO dataset
4. Create downloadable ZIP

Usage:
    python main.py
"""

from pathlib import Path
from src.duplicate_remover import DuplicateRemover
from src.augmentation import ImageAugmenter
from src.dataset_builder import YOLODatasetBuilder
from src.zip_creator import ZipCreator


def main():
    """
    Main function that runs the complete Phase 1 & 2 pipeline.
    """
    
    print("="*70)
    print("🚀 MINI DATASET GENERATOR - PHASE 1 & 2")
    print("="*70)
    
    # Setup paths
    current_dir = Path(__file__).parent
    input_folder = current_dir / "test_images"
    output_folder = current_dir / "output"
    
    # Check if input folder exists and has images
    if not input_folder.exists():
        print(f"\n❌ Error: Input folder not found!")
        print(f"   Please create folder: {input_folder}")
        print(f"   And add some images to it.")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend([str(f) for f in input_folder.glob(ext)])
    
    if not image_files:
        print(f"\n❌ Error: No images found in {input_folder}")
        print(f"   Please add some .jpg, .png, or .bmp images.")
        return
    
    print(f"\n📸 Found {len(image_files)} images in: {input_folder.name}")
    
    # =========================================================================
    # STEP 1: Remove Duplicates
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 1: REMOVING DUPLICATES")
    print("="*70)
    
    remover = DuplicateRemover(threshold=5)
    unique_images, duplicates = remover.find_duplicates(str(input_folder))
    
    print(f"\n📊 Duplicate Detection Results:")
    print(f"   Unique images: {len(unique_images)}")
    print(f"   Duplicates found: {len(duplicates)}")
    
    # Use only unique images for next steps
    working_images = unique_images
    
    # =========================================================================
    # STEP 2: Augment Images
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 2: AUGMENTING IMAGES")
    print("="*70)
    
    augmenter = ImageAugmenter()
    
    # Ask user how many augmentations per image
    print(f"\n💡 You have {len(working_images)} unique images")
    try:
        num_aug = int(input(f"   How many augmentations per image? (recommended: 5-10): "))
    except:
        num_aug = 5
        print(f"   Using default: {num_aug} augmentations per image")
    
    # Perform augmentation
    augmented_data = augmenter.augment_batch(working_images, num_augmentations=num_aug)
    
    # Save augmented images
    aug_output_folder = output_folder / "augmented_images"
    aug_output_folder.mkdir(parents=True, exist_ok=True)
    
    all_augmented_paths = []
    for original_path, aug_list in augmented_data.items():
        saved_paths = augmenter.save_augmented_images(aug_list, str(aug_output_folder))
        all_augmented_paths.extend(saved_paths)
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Original images: {len(working_images)}")
    print(f"   Augmented images: {len(all_augmented_paths)}")
    print(f"   Total dataset size: {len(working_images) + len(all_augmented_paths)}")
    
    # Combine original + augmented for dataset
    all_dataset_images = working_images + all_augmented_paths
    
    # =========================================================================
    # STEP 3: Build YOLO Dataset
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 3: BUILDING YOLO DATASET")
    print("="*70)
    
    # Ask for dataset name
    try:
        dataset_name = input(f"\n   Dataset name (default: my_dataset): ").strip()
        if not dataset_name:
            dataset_name = "my_dataset"
    except:
        dataset_name = "my_dataset"
    
    # Ask for class names
    try:
        classes_input = input(f"   Object classes (comma-separated, e.g. apple,orange): ").strip()
        if classes_input:
            class_names = [c.strip() for c in classes_input.split(',')]
        else:
            class_names = ['object']
    except:
        class_names = ['object']
    
    print(f"\n   Dataset name: {dataset_name}")
    print(f"   Classes: {', '.join(class_names)}")
    
    # Build the dataset
    builder = YOLODatasetBuilder(dataset_name=dataset_name)
    result = builder.build_dataset(
        image_paths=all_dataset_images,
        output_path=str(output_folder),
        class_names=class_names,
        train_ratio=0.8,
        create_labels=True
    )
    
    # =========================================================================
    # STEP 4: Create ZIP File (Phase 2)
    # =========================================================================
    print(f"\n" + "="*70)
    print("STEP 4: CREATING DOWNLOADABLE ZIP")
    print("="*70)
    
    zip_creator = ZipCreator()
    
    # Ask if user wants to include augmented images in ZIP
    try:
        include_aug = input(f"\n   Include augmented images folder in ZIP? (y/n, default: n): ").strip().lower()
        include_augmented = include_aug == 'y'
    except:
        include_augmented = False
    
    # Create the ZIP file
    try:
        if include_augmented:
            zip_file = zip_creator.create_dataset_zip(
                dataset_path=result['dataset_path'],
                include_augmented=True,
                augmented_path=str(aug_output_folder)
            )
        else:
            zip_file = zip_creator.create_dataset_zip(
                dataset_path=result['dataset_path'],
                include_augmented=False
            )
        
        # Get ZIP info
        zip_info = zip_creator.get_zip_info(zip_file)
        
        print(f"\n📊 ZIP File Statistics:")
        print(f"   📁 Files: {zip_info['file_count']}")
        print(f"   💾 Size: {zip_info['compressed_size_mb']:.2f} MB")
        print(f"   🗜️  Compression: {zip_info['compression_ratio']:.1f}%")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Could not create ZIP file: {e}")
        zip_file = None
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n" + "="*70)
    print("✅ PHASE 1 & 2 COMPLETE!")
    print("="*70)
    print(f"\n📦 Your dataset is ready:")
    print(f"   📁 Location: {result['dataset_path']}")
    print(f"   📄 Config: {result['yaml_path']}")
    print(f"   🎓 Training images: {result['train_images']}")
    print(f"   ✅ Validation images: {result['valid_images']}")
    print(f"   🏷️  Classes: {', '.join(result['classes'])}")
    
    if zip_file:
        print(f"\n📦 Download Ready:")
        print(f"   📥 ZIP file: {zip_file}")
        print(f"   💾 Size: {zip_info['compressed_size_mb']:.2f} MB")
    
    print(f"\n📚 Next Steps:")
    print(f"   1. Annotate your images using LabelImg or Roboflow")
    print(f"   2. Train YOLO model using the generated data.yaml")
    print(f"   3. Move to Phase 3: Build Streamlit UI")
    print(f"\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
