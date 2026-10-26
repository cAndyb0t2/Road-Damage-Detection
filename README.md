# 🛣️ Road Damage Detection Using Deep Learning

## 📌 Overview

Road Damage Detection is a computer vision project that uses deep learning to identify potential road damage from images. The system uses a trained **YOLOv8 object detection model** to locate damage regions and displays the predictions through an interactive **Streamlit web application**.

The application allows users to upload road images, visualize detected regions using bounding boxes, review confidence scores, and download the annotated images and detection reports.

## 🎯 Objectives

- Automate the identification of potential road damage from images.
- Apply YOLOv8 for object detection.
- Develop a simple and interactive web interface using Streamlit.
- Display predicted bounding boxes and confidence scores.
- Export detection results for further analysis.

## ✨ Features

- 📤 Upload road images in JPG, JPEG, and PNG formats.
- 🤖 Detect potential road damage using a trained YOLOv8 model.
- 📦 Visualize predictions with bounding boxes.
- 📊 View detection counts and confidence scores.
- ⚙️ Adjust the confidence threshold.
- 📥 Download annotated images and CSV detection reports.
- 🕘 View recent analyses during the current session.

## 🧰 Technologies Used

- **Python** — Core programming language
- **YOLOv8** — Object detection model
- **PyTorch** — Deep learning framework
- **Ultralytics** — YOLO model implementation
- **Streamlit** — Web application interface
- **Pillow** — Image handling
- **Pandas** — Detection report generation

## 🏗️ System Workflow

1. Upload a road image through the web interface.
2. Pass the image to the trained YOLOv8 model.
3. Perform object detection and obtain predicted bounding boxes.
4. Filter predictions using the selected confidence threshold.
5. Display the annotated image and detection statistics.
6. Download the output image and detection report.

## 📂 Project Structure

```text
RoadDamageDetection/
├── Models/
│   └── best.pt
├── app.py
├── road.jpg
└── README.md
```

*The project structure may vary depending on the local setup. The trained model file is required to run the application.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RoadDamageDetection
```

### 2. Install dependencies

```bash
python -m pip install streamlit ultralytics pillow pandas
```

### 3. Configure the model path

Ensure that the trained `best.pt` model is available in the `Models` directory. Update the model path in `app.py` if your directory structure differs.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 🧪 Model Information

- **Model:** YOLOv8n
- **Task:** Object detection
- **Number of classes:** 5
- **Training dataset:** RDD2022
- **Model weights:** `best.pt`

The model was trained using the RDD2022 road damage dataset. Its predictions depend on the training data, image quality, and confidence threshold. The class names should be mapped to the verified dataset labels before interpreting individual damage categories.

## 🚀 Future Enhancements

- Map model class IDs to verified road damage categories.
- Improve detection accuracy through further model evaluation and training.
- Add video-based road damage detection.
- Integrate GPS-based location tagging.
- Develop a dashboard for road inspection and maintenance reporting.

## ⚠️ Limitations

The system provides predictions based on a trained model and may produce missed detections or false positives. Results should be reviewed before being used for road maintenance decisions.

## 👨‍💻 Applications

- Road inspection assistance
- Road maintenance planning
- Infrastructure monitoring
- Smart city research
- Computer vision and deep learning projects



**Developed as an academic project on deep learning and computer vision.**
