# Celebrity Face Detection using YOLOv8

This project demonstrates how to train a YOLOv8 model for celebrity face detection using a custom dataset from Roboflow. The model is capable of identifying and localizing celebrity faces in images.

![sharapova_labelled](https://github.com/user-attachments/assets/1637650c-7c8e-403f-ac73-87880385abb9)


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Project Link](#project-link)

## Overview

This project uses the YOLOv8 object detection model to detect celebrity faces in images. The model is trained on a custom dataset created and annotated using Roboflow.

## Installation

To set up the project, you need to install the required dependencies:

```bash
pip install ultralytics roboflow
```

## Dataset

The dataset used in this project is a custom celebrity face detection dataset created and annotated on Roboflow. To access the dataset:

1. Sign up for a Roboflow account and obtain an API key.
2. Use the following code to download the dataset:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR-API-KEY")
project = rf.workspace("puspendu-ai-vision-workspace").project("celebrity-face-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

3. You can also download the dataset locally on your machine by visiting the project [here](https://universe.roboflow.com/puspendu-ai-vision-workspace/celebrity-face-detection)

The dataset is organized into train, validation, and test splits, ready for use with YOLOv8.

## Model Training

To train the YOLOv8 model on the celebrity face detection dataset:

```python
from ultralytics import YOLO

model = YOLO("yolov8l.pt")  # Initialize with pre-trained weights
model.train(
    data="Celebrity-Face-Detection-1/data.yaml",
    epochs=50,
    imgsz=640
)
```

The trained model will be saved in the `runs/detect/train` directory.

## Usage

After training, you can use the model to detect celebrity faces in new images:

```python
def display_prediction(image_path, save_fig = False, filename = None):
    """
    Function to display predictions on a given image using our trained model.
    
    Args:
    image_path (str): Path to the input image
    """
    # Load the trained model
    trained_model = YOLO("runs/detect/train4/weights/best.pt")
    
    # Make predictions
    result = trained_model.predict(source=image_path, conf = 0.5)
    
    # Convert the result to RGB for display
    rgb_image = cv.cvtColor(result[0].plot(), cv.COLOR_BGR2RGB)
    
    # Display the result
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.axis('off')

    save_dir = 'DATA/predicted_images'
    if save_fig and filename:
        file_path = os.path.join(save_dir, filename)
        fig.savefig(file_path, dpi = 200, bbox_inches='tight')
    plt.show()
```

## Results

The model has been tested on various celebrity images and demonstrates promising results in detecting and localizing celebrity faces. You can find example outputs in the notebook. I have included some prediction results below.

![Virat_labelled](https://github.com/user-attachments/assets/0d30f6c1-49b3-497c-8242-91fb7879ff32)

![katherine_labelled](https://github.com/user-attachments/assets/a7332743-55db-4155-ba88-143f29f69568)


## Future Improvements

To enhance the model's performance, consider:

1. Experimenting with different YOLOv8 variants (nano, small, medium, large, extra-large)
2. Adjusting training parameters (learning rate, batch size, etc.)
3. Augmenting the dataset to increase its size and variety
4. Fine-tuning the model on a larger, more diverse dataset of celebrity faces

## Project Link

You can check out the project on Roboflow by clicking [here](https://universe.roboflow.com/puspendu-ai-vision-workspace/celebrity-face-detection)

---
