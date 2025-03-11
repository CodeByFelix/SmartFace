# SmartFace Library

SmartFace is a Python library for real-time face emotion analysis using a Convolutional Neural Network (CNN) model trained on the AffectNet dataset. It detects faces in images and video streams using OpenCV's Haar cascade classifier and predicts emotions using a pre-trained deep learning model.

## Features
- Detects faces in images and videos
- Predicts emotions such as Angry, Happy, Sad, Neutral, etc.
- Draws bounding boxes and labels emotions on detected faces
- Supports real-time emotion detection via webcam

## Installation

First, clone this repository:

```bash
git clone https://github.com/CodeByFelix/SmartFace.git
cd SmartFace
```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Import the SmartFace class and use it for emotion detection.

### Running on Video
```python
from smartface import SmartFace
import cv2

# Load the model
modelPath = "Face_Emotion_Model.h5" # Make sure to include the model to your project directory or provide the path to the model
smartFace = SmartFace(modelPath)

# Open webcam for real-time emotion detection
video = cv2.VideoCapture(0)
smartFace.stream(video)
```

### Running on Image
```python
import cv2
from smartface import SmartFace

modelPath = "Face_Emotion_Model.h5" # Make sure to include the model to your project directory or provide the path to the model
smartFace = SmartFace(modelPath)

# Load an image
image = cv2.imread("path/to/image.jpg")
result = smartFace.analyze(image)

# Print detected emotions
print(result)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Feel free to submit issues or pull requests to improve the project!

## Author
- **Author**: Felix Ibeamaka 
- 📧 **Email**: felixibeamaka123@gmail.com  
- 📞 **Phone**: +2347037872133
