# 🎨 Mini Dataset Generator

A web-based tool to create YOLO-ready datasets from a small set of images using augmentation and intelligent duplicate removal.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## ✨ Features

- **🔍 Smart Duplicate Detection** - Uses perceptual hashing to find and remove similar images
- **🎨 Image Augmentation** - Automatically increases dataset size with realistic variations
- **📦 YOLO Dataset Builder** - Creates proper folder structure (train/test/valid) for YOLOv11
- **📊 3-Way Split** - Automatically splits data into training, test, and validation sets
- **🌐 Web Interface** - Easy-to-use Streamlit UI with drag-and-drop upload
- **📥 One-Click Download** - Get your dataset as a ZIP file

## 🚀 Quick Start (Web App)

### Option 1: Use Online (Recommended)
Visit the deployed app: [Mini Dataset Generator](https://your-app-url.streamlit.app)

### Option 2: Run Locally
```powershell
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🎯 How to Use

1. **Upload Images** - Drag and drop 10-20 images
2. **Configure Settings** - Adjust augmentation, split ratios in sidebar
3. **Generate Dataset** - Click the "Generate Dataset" button
4. **Download ZIP** - Get your YOLO-ready dataset

## 📁 Project Structure

```
mini_dataset_generator/
├── src/
│   ├── duplicate_remover.py    # Perceptual hashing duplicate detection
│   ├── augmentation.py          # Image augmentation engine
│   ├── dataset_builder.py       # YOLO folder structure generator
│   └── zip_creator.py           # ZIP file creation
├── app.py                       # Streamlit web application
├── main.py                      # CLI pipeline script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🎓 Learning Mode

This project is designed for learning! Each module includes:
- ✅ Detailed comments explaining every function
- ✅ Docstrings with parameter descriptions
- ✅ Example usage at the bottom of each file
- ✅ Step-by-step explanations

**Read the [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for:**
- How each algorithm works
- Key concepts explained
- Code walkthroughs
- Exercises to test your understanding

## 🔧 How It Works

### Pipeline Overview:
```
Input Images → Duplicate Removal → Augmentation → YOLO Dataset → Ready for Training
```

### Detailed Steps:

1. **Duplicate Detection**
   - Uses perceptual hashing (not MD5/SHA)
   - Finds similar images even with slight variations
   - Configurable similarity threshold

2. **Image Augmentation**
   - Rotation, flipping, scaling
   - Brightness/contrast adjustments
   - Color shifts, blur, noise
   - Perspective changes
   - Increases dataset from 10 → 100+ images

3. **YOLO Dataset Structure**
   ```
   my_dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

## 🧪 Test Individual Modules

Each module can be tested independently:

```powershell
# Test duplicate remover
python src/duplicate_remover.py

# Test augmentation
python src/augmentation.py

# Test dataset builder
python src/dataset_builder.py
```

## 📦 Example Output

**Input:** 15 images in `test_images/`

**After Processing:**
- Duplicates removed: 3 (12 unique remain)
- Augmentation (5x): 12 × 5 = 60 new images
- Total dataset: 72 images
- Train set: 58 images (80%)
- Valid set: 14 images (20%)

## 🛠 Requirements

```
opencv-python >= 4.8.0
Pillow >= 10.0.0
imagehash >= 4.3.1
numpy >= 1.24.0
albumentations >= 1.3.1
ultralytics >= 8.0.0
PyYAML >= 6.0
```

## 🎯 Use Cases

- **Small Dataset Expansion** - Turn 10 images into 100+
- **YOLO Training Prep** - Creates proper folder structure
- **Quick Prototyping** - Test CV ideas without collecting huge datasets
- **Learning Tool** - Understand CV pipelines and data preprocessing

## 🚀 Next Steps (Phase 2)

Phase 2 will add:
- 🌐 **Streamlit Web UI** - Upload images via browser
- 📥 **Direct Download** - Download dataset as ZIP
- 👁️ **Image Preview** - See augmentations before building
- ⚙️ **Parameter Controls** - Adjust settings with sliders

## 🐛 Troubleshooting

### "Module not found"
```powershell
pip install -r requirements.txt
```

### "No images found"
Add .jpg, .png, or .bmp images to `test_images/` folder

### Images look distorted
Adjust augmentation parameters in `src/augmentation.py` - reduce rotation angles or brightness changes

## 📚 What You'll Learn

- **Computer Vision:** Perceptual hashing, image transforms, augmentation
- **Python:** OOP, Pathlib, type hints, error handling
- **ML/AI:** YOLO format, train/valid splits, data pipelines
- **Best Practices:** Code organization, documentation, testing

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new augmentation transforms
- Improve the duplicate detection algorithm
- Add more dataset format support (COCO, Pascal VOC)
- Create better visualizations

## 📄 License

Free to use for learning and personal projects.

## 💡 Tips

- Start with 10-15 images for testing
- Use 5-10 augmentations per image
- Check augmented images before training
- Label files are empty - use LabelImg or Roboflow to annotate

## 🎓 Learning Resources

- [LEARNING_GUIDE.md](LEARNING_GUIDE.md) - Complete tutorial
- Comments in each Python file
- Docstrings for every function
- Example usage at bottom of modules

---

**Happy Learning! 🚀**

*Built as a learning project for Computer Vision and Python development*
