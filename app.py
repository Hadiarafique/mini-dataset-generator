"""
Streamlit Web Application - Phase 3 & 4
========================================
A web-based UI for the Mini Dataset Generator.

Features:
- Upload images via drag-and-drop
- Configure processing parameters
- Preview uploaded images
- Process dataset with progress tracking
- Download ZIP file
- Error handling and validation
- Statistics dashboard

Usage:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import time
import json
from datetime import datetime

from src.duplicate_remover import DuplicateRemover
from src.augmentation import ImageAugmenter
from src.dataset_builder import YOLODatasetBuilder
from src.zip_creator import ZipCreator


# Page configuration
st.set_page_config(
    page_title="Mini Dataset Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def validate_image(file) -> tuple:
    """Validate uploaded image file."""
    try:
        if file.size > 10 * 1024 * 1024:
            return False, f"{file.name}: File too large (max 10MB)"
        
        img = Image.open(file)
        
        if img.width < 50 or img.height < 50:
            return False, f"{file.name}: Image too small (min 50x50 pixels)"
        
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False, f"{file.name}: Unsupported image mode"
        
        return True, None
    except Exception as e:
        return False, f"{file.name}: Invalid image file - {str(e)}"


def load_settings():
    """Load saved settings from file."""
    settings_file = Path("settings.json")
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'dataset_name': 'my_dataset',
        'class_names': 'object',
        'duplicate_threshold': 5,
        'num_augmentations': 5,
        'train_ratio': 0.7,
        'test_ratio': 0.15
    }


def save_settings(settings):
    """Save settings to file."""
    try:
        with open("settings.json", 'w') as f:
            json.dump(settings, f)
    except:
        pass


def main():
    """Main application function."""
    
    # Initialize session state
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    
    # Header
    st.markdown('<p class="main-header">🎨 Mini Dataset Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create professional YOLO datasets with AI-powered augmentation and duplicate removal</p>', unsafe_allow_html=True)
    
    st.info("💡 **Quick Start:** Upload images → Configure settings in sidebar → Click Generate Dataset → Download ZIP")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📦 Dataset Settings")
    dataset_name = st.sidebar.text_input(
        "Dataset Name",
        value=st.session_state.settings.get('dataset_name', 'my_dataset'),
        help="Name for your YOLO dataset"
    )
    
    class_names_input = st.sidebar.text_input(
        "Object Classes",
        value=st.session_state.settings.get('class_names', 'object'),
        help="Comma-separated list (e.g., apple,orange,banana)"
    )
    class_names = [c.strip() for c in class_names_input.split(',') if c.strip()]
    
    if not class_names:
        st.sidebar.warning("⚠️ At least one class name is required")
        class_names = ['object']
    
    st.sidebar.subheader("🏷️ Label Support")
    has_labels = st.sidebar.checkbox(
        "I have existing labels",
        value=False,
        help="Check if you want to upload labels with your images"
    )
    
    export_format = st.sidebar.selectbox(
        "Export Label Format",
        options=["YOLO", "COCO JSON", "Pascal VOC XML"],
        index=0,
        help="Format for exported labels"
    )
    
    st.sidebar.subheader("🔍 Duplicate Detection")
    duplicate_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0,
        max_value=15,
        value=st.session_state.settings.get('duplicate_threshold', 5),
        help="Lower = stricter (0=exact match only, 15=very lenient)"
    )
    
    st.sidebar.subheader("🎨 Augmentation")
    num_augmentations = st.sidebar.slider(
        "Augmentations per Image",
        min_value=1,
        max_value=20,
        value=st.session_state.settings.get('num_augmentations', 5),
        help="How many augmented versions to create for each image"
    )
    
    if 'uploaded_file_count' in st.session_state:
        expected_output = st.session_state.uploaded_file_count * (num_augmentations + 1)
        st.sidebar.caption(f"📊 Expected output: ~{expected_output} images")
    
    st.sidebar.subheader("✂️ Train/Test/Valid Split")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        train_ratio = st.slider(
            "Train %",
            min_value=50,
            max_value=85,
            value=int(st.session_state.settings.get('train_ratio', 0.7) * 100),
            step=5,
            help="Percentage for training"
        ) / 100
    
    with col2:
        test_ratio = st.slider(
            "Test %",
            min_value=5,
            max_value=30,
            value=int(st.session_state.settings.get('test_ratio', 0.15) * 100),
            step=5,
            help="Percentage for testing"
        ) / 100
    
    valid_ratio = 1.0 - train_ratio - test_ratio
    if valid_ratio < 0:
        valid_ratio = 0.05
        train_ratio = 1.0 - test_ratio - valid_ratio
    
    st.sidebar.caption(f"📊 Split: {int(train_ratio*100)}% train / {int(test_ratio*100)}% test / {int(valid_ratio*100)}% valid")
    
    st.sidebar.subheader("📦 ZIP Options")
    include_augmented = st.sidebar.checkbox(
        "Include augmented images in ZIP",
        value=False,
        help="Add the augmented_images folder to the ZIP file"
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Save Settings as Default"):
        st.session_state.settings = {
            'dataset_name': dataset_name,
            'class_names': class_names_input,
            'duplicate_threshold': duplicate_threshold,
            'num_augmentations': num_augmentations,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio
        }
        save_settings(st.session_state.settings)
        st.sidebar.success("✅ Settings saved!")
    
    with st.sidebar.expander("❓ Help & Tips"):
        st.markdown("""
        **Recommended Settings:**
        - Images: Minimum 10, 50+ for production
        - Augmentations: 5-10 per image
        - Duplicate threshold: 5 (default)
        - Train/Test/Valid: 70/15/15 (standard)
        
        **Image Requirements:**
        - Format: JPG, PNG, BMP
        - Min size: 50x50 pixels
        - Max size: 10MB per file
        """)
    
    # Main content
    st.markdown("---")
    
    # File uploader
    st.subheader("📸 Step 1: Upload Images")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload your images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Drag and drop images here or click to browse"
        )
    with col2:
        st.markdown("**Image Stats:**")
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.metric("Files", len(uploaded_files))
            st.metric("Total Size", f"{total_size:.1f} MB")
    
    # Label upload (optional)
    uploaded_labels = None
    if has_labels:
        st.subheader("🏷️ Step 1b: Upload Labels (Optional)")
        uploaded_labels = st.file_uploader(
            "Upload YOLO label files (.txt)",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload .txt label files matching your image names"
        )
        
        if uploaded_labels:
            st.success(f"✅ Uploaded {len(uploaded_labels)} label files")
            st.info("💡 **Tip:** Label filenames should match image filenames (e.g., `image1.jpg` → `image1.txt`)")
    
    if uploaded_files:
        st.session_state.uploaded_file_count = len(uploaded_files)
        
        # Validate images
        invalid_files = []
        for file in uploaded_files:
            is_valid, error_msg = validate_image(file)
            if not is_valid:
                invalid_files.append(error_msg)
        
        if invalid_files:
            st.error("❌ Some files failed validation:")
            for error in invalid_files:
                st.text(f"  • {error}")
            st.warning("Please remove invalid files and try again.")
            return
        
        st.success(f"✅ All {len(uploaded_files)} images validated successfully")
        
        # Preview images
        with st.expander("👁️ Preview Uploaded Images", expanded=False):
            cols = st.columns(5)
            for idx, uploaded_file in enumerate(uploaded_files[:10]):
                with cols[idx % 5]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
            
            if len(uploaded_files) > 10:
                st.info(f"Showing first 10 of {len(uploaded_files)} images")
        
        # Processing button
        st.markdown("---")
        st.subheader("⚡ Step 2: Process Dataset")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Generate Dataset",
                type="primary",
                use_container_width=True
            )
        
        if process_button:
            process_dataset(
                uploaded_files=uploaded_files,
                dataset_name=dataset_name,
                class_names=class_names,
                duplicate_threshold=duplicate_threshold,
                num_augmentations=num_augmentations,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                valid_ratio=valid_ratio,
                include_augmented=include_augmented,
                uploaded_labels=uploaded_labels,
                export_format=export_format
            )
    else:
        st.info("👆 Upload images to get started")
        
        # Show example
        st.markdown("---")
        st.subheader("💡 How it works")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 1️⃣ Upload")
            st.write("Upload images of your objects")
        
        with col2:
            st.markdown("### 2️⃣ Remove Duplicates")
            st.write("AI detects and removes similar images")
        
        with col3:
            st.markdown("### 3️⃣ Augment")
            st.write("Creates variations (rotate, flip, etc.)")
        
        with col4:
            st.markdown("### 4️⃣ Download")
            st.write("Get YOLO-ready dataset as ZIP")


def process_dataset(uploaded_files, dataset_name, class_names, duplicate_threshold, 
                   num_augmentations, train_ratio, test_ratio, valid_ratio, include_augmented,
                   uploaded_labels=None, export_format="YOLO"):
    """Process uploaded images and create dataset with optional label support."""
    
    start_time = time.time()
    
    try:
        # Use persistent output folder
        current_dir = Path(__file__).parent
        output_folder = current_dir / "output"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Clean up old temporary folders and datasets from previous sessions
        for item in output_folder.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)  # Remove all old folders
                except Exception as e:
                    st.warning(f"Could not clean up {item.name}: {str(e)}")
        
        # Also remove old ZIP files to save space
        for zip_file in output_folder.glob("*.zip"):
            try:
                zip_file.unlink()
            except Exception as e:
                pass
        
        # Create fresh input folder
        input_folder = output_folder / "temp_input_images"
        input_folder.mkdir(parents=True, exist_ok=True)
        
        # Create labels folder if labels are provided
        if uploaded_labels:
            input_labels_folder = output_folder / "temp_input_labels"
            input_labels_folder.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        st.info("💾 Saving uploaded images...")
        progress_bar = st.progress(0)
        
        image_paths = []
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                file_path = input_folder / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(str(file_path))
                progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"❌ Failed to save {uploaded_file.name}: {str(e)}")
                return
        
        st.success(f"✅ Saved {len(image_paths)} images")
        
        # Save uploaded labels if provided
        label_mapping = {}
        if uploaded_labels:
            st.info("💾 Saving uploaded labels...")
            progress_bar_labels = st.progress(0)
            
            for idx, uploaded_label in enumerate(uploaded_labels):
                try:
                    label_path = input_labels_folder / uploaded_label.name
                    with open(label_path, 'wb') as f:
                        f.write(uploaded_label.getbuffer())
                    
                    # Map label to image (by filename stem)
                    label_stem = Path(uploaded_label.name).stem
                    label_mapping[label_stem] = str(label_path)
                    progress_bar_labels.progress((idx + 1) / len(uploaded_labels))
                except Exception as e:
                    st.warning(f"⚠️ Failed to save label {uploaded_label.name}: {str(e)}")
            
            st.success(f"✅ Saved {len(label_mapping)} label files")
            
            # Validate label-image matching
            matched = 0
            for img_path in image_paths:
                img_stem = Path(img_path).stem
                if img_stem in label_mapping:
                    matched += 1
            
            if matched > 0:
                st.info(f"🔗 Matched {matched}/{len(image_paths)} images with labels")
            else:
                st.warning("⚠️ No labels matched with image filenames. Labels should have the same name as images.")
        
        # Step 1: Remove duplicates
        st.markdown("---")
        st.subheader("🔍 Step 1: Removing Duplicates")
        
        try:
            with st.spinner("Scanning for duplicates..."):
                remover = DuplicateRemover(threshold=duplicate_threshold)
                unique_images, duplicates = remover.find_duplicates(str(input_folder))
        except Exception as e:
            st.error(f"❌ Duplicate detection failed: {str(e)}")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Images", len(image_paths))
        with col2:
            st.metric("Unique Images", len(unique_images))
        with col3:
            st.metric("Duplicates Removed", len(duplicates))
        
        if len(duplicates) > 0:
            with st.expander("View Duplicate Files"):
                for dup in duplicates:
                    st.text(f"❌ {Path(dup).name}")
        
        if len(unique_images) == 0:
            st.error("❌ No unique images found. Please upload different images.")
            return
        
        # Step 2: Augmentation
        st.markdown("---")
        st.subheader("🎨 Step 2: Augmenting Images")
        
        try:
            augmenter = ImageAugmenter()
            aug_output_folder = output_folder / "augmented_images"
            aug_output_folder.mkdir(parents=True, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_augmented_paths = []
            total_images = len(unique_images)
            
            for idx, img_path in enumerate(unique_images):
                status_text.text(f"Processing {idx + 1}/{total_images}: {Path(img_path).name}")
                
                try:
                    augmented = augmenter.augment_image(img_path, num_augmentations)
                    saved_paths = augmenter.save_augmented_images(augmented, str(aug_output_folder))
                    all_augmented_paths.extend(saved_paths)
                except Exception as e:
                    st.warning(f"⚠️ Augmentation failed for {Path(img_path).name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / total_images)
            
            status_text.empty()
        except Exception as e:
            st.error(f"❌ Augmentation failed: {str(e)}")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Images", len(unique_images))
        with col2:
            st.metric("Augmented Images", len(all_augmented_paths))
        with col3:
            st.metric("Total Dataset Size", len(unique_images) + len(all_augmented_paths))
        
        # Preview augmented images
        with st.expander("👁️ Preview Augmented Images"):
            cols = st.columns(5)
            preview_count = min(10, len(all_augmented_paths))
            for idx in range(preview_count):
                with cols[idx % 5]:
                    try:
                        img = Image.open(all_augmented_paths[idx])
                        st.image(img, caption=Path(all_augmented_paths[idx]).name, use_container_width=True)
                    except:
                        st.error("Failed to load")
            if len(all_augmented_paths) > 10:
                st.info(f"Showing first 10 of {len(all_augmented_paths)} augmented images")
        
        # Combine all images
        all_dataset_images = unique_images + all_augmented_paths
        
        # Step 3: Build YOLO dataset
        st.markdown("---")
        st.subheader("📦 Step 3: Building YOLO Dataset")
        
        try:
            with st.spinner("Creating YOLO folder structure..."):
                builder = YOLODatasetBuilder(dataset_name=dataset_name)
                result = builder.build_dataset(
                    image_paths=all_dataset_images,
                    output_path=str(output_folder),
                    class_names=class_names,
                    train_ratio=train_ratio,
                    test_ratio=test_ratio,
                    valid_ratio=valid_ratio,
                    create_labels=True
                )
            
            # Verify images were copied
            dataset_path = Path(result['dataset_path'])
            train_img_count = len(list((dataset_path / 'train' / 'images').glob('*')))
            test_img_count = len(list((dataset_path / 'test' / 'images').glob('*')))
            valid_img_count = len(list((dataset_path / 'valid' / 'images').glob('*')))
            
            if train_img_count == 0 and test_img_count == 0 and valid_img_count == 0:
                st.error("❌ No images were copied to dataset folders. Please check the logs.")
                return
                
        except Exception as e:
            st.error(f"❌ Dataset building failed: {str(e)}")
            st.exception(e)
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Images", result['train_images'])
        with col2:
            st.metric("Test Images", result['test_images'])
        with col3:
            st.metric("Validation Images", result['valid_images'])
        with col4:
            st.metric("Classes", len(result['classes']))
        
        st.info(f"🏷️ Classes: {', '.join(result['classes'])}")
        
        # Step 3.5: Convert label formats if requested
        if export_format != "YOLO" and uploaded_labels:
            st.markdown("---")
            st.subheader(f"📄 Step 3.5: Converting Labels to {export_format}")
            
            try:
                from src.label_converter import LabelConverter
                
                with st.spinner(f"Converting labels to {export_format}..."):
                    converter = LabelConverter(class_names=class_names)
                    dataset_path = Path(result['dataset_path'])
                    
                    if export_format == "COCO JSON":
                        # Convert train, test, valid separately
                        for split in ['train', 'test', 'valid']:
                            images_dir = dataset_path / split / 'images'
                            labels_dir = dataset_path / split / 'labels'
                            output_json = dataset_path / split / f'annotations_{split}.json'
                            
                            if images_dir.exists() and labels_dir.exists():
                                converter.yolo_to_coco(str(images_dir), str(labels_dir), str(output_json))
                        
                        st.success(f"✅ Converted labels to COCO JSON format")
                    
                    elif export_format == "Pascal VOC XML":
                        # Convert all splits
                        for split in ['train', 'test', 'valid']:
                            images_dir = dataset_path / split / 'images'
                            labels_dir = dataset_path / split / 'labels'
                            xml_dir = dataset_path / split / 'annotations_xml'
                            xml_dir.mkdir(exist_ok=True)
                            
                            if images_dir.exists() and labels_dir.exists():
                                for img_file in images_dir.glob('*'):
                                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                        label_file = labels_dir / f"{img_file.stem}.txt"
                                        if label_file.exists():
                                            converter.yolo_to_voc(str(img_file), str(label_file), str(xml_dir))
                        
                        st.success(f"✅ Converted labels to Pascal VOC XML format")
                
            except Exception as e:
                st.warning(f"⚠️ Label conversion failed: {str(e)}")
        
        # Step 4: Create ZIP
        st.markdown("---")
        st.subheader("📦 Step 4: Creating ZIP File")
        
        try:
            with st.spinner("Compressing dataset..."):
                zip_creator = ZipCreator()
                
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
                
                zip_info = zip_creator.get_zip_info(zip_file)
                st.success(f"✅ ZIP saved to: {zip_file}")
                
        except Exception as e:
            st.error(f"❌ ZIP creation failed: {str(e)}")
            st.exception(e)
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files in ZIP", zip_info['file_count'])
        with col2:
            st.metric("ZIP Size", f"{zip_info['compressed_size_mb']:.2f} MB")
        with col3:
            st.metric("Compression", f"{zip_info['compression_ratio']:.1f}%")
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Download button
        st.markdown("---")
        st.success(f"✅ Dataset Generation Complete! (Processing time: {processing_time:.1f}s)")
        
        with open(zip_file, 'rb') as f:
            zip_data = f.read()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Dataset ZIP",
                data=zip_data,
                file_name=Path(zip_file).name,
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
        
        st.info(f"💾 ZIP file is also saved at: `{zip_file}`")
        
        # Final summary
        st.markdown("---")
        st.subheader("📊 Processing Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("**Dataset Information:**")
            st.write(f"• Name: `{dataset_name}`")
            st.write(f"• Classes: `{', '.join(class_names)}`")
            st.write(f"• Total Images: `{result['train_images'] + result['test_images'] + result['valid_images']}`")
            st.write(f"• Train/Test/Valid Split: `{int(train_ratio*100)}/{int(test_ratio*100)}/{int(valid_ratio*100)}`")
            st.write(f"• Processing Time: `{processing_time:.1f}s`")
            st.write(f"• ZIP Size: `{zip_info['compressed_size_mb']:.2f} MB`")
            st.write(f"• Location: `{output_folder}`")
        
        with summary_col2:
            st.markdown("**Next Steps:**")
            st.write("1. ✅ Download the ZIP file above")
            st.write("2. 📂 Extract the dataset")
            st.write("3. 🏷️ Annotate images using LabelImg or Roboflow")
            st.write("4. 🚀 Train YOLO model with `data.yaml`")
            st.write("5. 🎯 Test and deploy your model")
        
        # Statistics
        with st.expander("📈 Detailed Statistics"):
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.markdown("**Processing Stats:**")
                st.write(f"• Uploaded: {len(uploaded_files)} files")
                st.write(f"• Duplicates: {len(duplicates)} removed")
                st.write(f"• Unique: {len(unique_images)} images")
                st.write(f"• Augmented: {len(all_augmented_paths)} created")
            
            with stat_col2:
                st.markdown("**Dataset Stats:**")
                st.write(f"• Training: {result['train_images']} images")
                st.write(f"• Test: {result['test_images']} images")
                st.write(f"• Validation: {result['valid_images']} images")
                st.write(f"• Total: {result['total_images']} images")
                st.write(f"• Classes: {len(class_names)} defined")
            
            with stat_col3:
                st.markdown("**File Stats:**")
                st.write(f"• ZIP files: {zip_info['file_count']}")
                st.write(f"• Compressed: {zip_info['compressed_size_mb']:.2f} MB")
                st.write(f"• Uncompressed: {zip_info['uncompressed_size_mb']:.2f} MB")
                st.write(f"• Savings: {zip_info['compression_ratio']:.1f}%")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
