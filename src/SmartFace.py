"""
SmartFace Library
=================
This library enables face emotion analysis using a CNN model trained on the AffectNet dataset.
It detects faces in images and video streams using OpenCV's Haar cascade classifier
and predicts emotions using a pre-trained deep learning model.

Author: Felix Ibeamaka
Version: 1.0.0
"""


import cv2
import numpy as np
import tensorflow as tf


class SmartFace:
    def __init__ (self, modelPath: str = "models/Face_Emotion_Model.h5"):
        """
        Initializes the SmartFace class.

        Args:
            model_path (str): Path to the pre-trained model file.
        """
        self.faceEmotion = tf.keras.models.load_model(modelPath)
        self.cascadePath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.cascadePath)
    
    def _imagePreprocess (self, image: np.ndarray) -> np.ndarray :
        """
        Preprocesses an image for model inference.

        Args:
            image (np.ndarray): Input image of a detected face.

        Returns:
            np.ndarray: Processed image ready for model input.
        """
        image = cv2.resize (image, (48, 48))
        image = image.astype ('float32')
        image = image/255.0
        image = np.reshape (image, (1, 48, 48, 3))
        return image
    
    def _predictEmotion (self, image: np.ndarray) -> tuple[dict, str]:
       """
        Predicts emotion from a preprocessed face image.

        Args:
            image (np.ndarray): Preprocessed image.

        Returns:
            tuple: Dictionary of emotion probabilities and the dominant emotion label.
        """
       pred = self.faceEmotion.predict (image)
       emotionList = ['Angry', 'Disgust', 'Fear', 'Happy', "Neutral", 'Sad', 'Surprise']
       emotion = {emotionList[i]: round(pred[0][i]*100.0, 2) for i in range(len(emotionList))}
       dominant_emotion = emotionList[np.argmax(pred)]
       return emotion, dominant_emotion
    
    def _detectFaces (self, image: np.ndarray) -> list:
        """
        Detects faces in an image using OpenCV's Haar cascade classifier.

        Args:
            image (np.ndarray): Input image.

        Returns:
            list: List of detected faces with bounding boxes.
        """
        faceImages =[]
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale (imageGray, scaleFactor=1.1, minNeighbors=4, minSize=(28, 28))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            faceImages.append([face, [x, y, w, h]])
        
        return faceImages
        
    def analyze (self, image: np.ndarray) -> list[dict]:
        """
        Detects faces and predicts emotions for each face in an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            list: List of dictionaries containing emotions and bounding box data.
        """
        faceImages = self._detectFaces(image)
        
        faceEmotion =[]
        for face, bbox in faceImages:
            img = self._imagePreprocess(face)
            emotion, dominant_emotion = self._predictEmotion (img)
            emo = {
                'emotions': emotion,
                'dominant_emotion': dominant_emotion,
                'face':{'x':bbox[0], 'y':bbox[1], 'w':bbox[2], 'h':bbox[3]}
                }
            faceEmotion.append(emo)
            
        return faceEmotion
    
    def analyzeDraw (self, image: np.ndarray) -> np.ndarray:
        """
        Detects faces, predicts emotions, and draws results on the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with bounding boxes and emotions drawn.
        """
        faceImages = self._detectFaces(image)
        
        for face, bbox in faceImages:
            img = self._imagePreprocess(face)
            emotion, dominant_emotion = self._predictEmotion (img)
            
            x, y, w, h = bbox
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontScale = 0.5
            fontThickness = 1
            textSize = cv2.getTextSize (dominant_emotion, font, fontScale, fontThickness)[0]
            
            if y <= 5.0:
                t_tl = (x, y+textSize[1]+5)
                t_br = (x+textSize[0], y) 
                cv2.rectangle(image, (x, y, w, h), (255, 0, 255), 2)
                cv2.rectangle(image, t_tl, t_br, (255, 0, 255), cv2.FILLED)
                cv2.putText (image, dominant_emotion, (x, y+textSize[1]), font, fontScale, (255, 255, 255), fontThickness)
                
            
            else:
                t_tl = (x, y-textSize[1]-5)
                t_br = (x+textSize[0], y) 
                cv2.rectangle(image, (x, y, w, h), (255, 0, 255), 2)
                cv2.rectangle(image, t_tl, t_br, (255, 0, 255), cv2.FILLED)
                cv2.putText (image, dominant_emotion, (x, y-textSize[1]), font, fontScale, (255, 255, 255), fontThickness)
                
            
        return image
            
    def stream (self, video: cv2.VideoCapture):
        """
        Opens a video stream and applies real-time face emotion detection.

        Args:
            video (cv2.VideoCapture): OpenCV VideoCapture object.
        """
        while video.isOpened():
            ret, frame = video.read()
            
            if ret:
                faceImages = self._detectFaces(frame)
                
                for face, bbox in faceImages:
                    img = self._imagePreprocess(face)
                    emotion, dominant_emotion = self._predictEmotion (img)
                    
                    x, y, w, h = bbox
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    fontScale = 0.5
                    fontThickness = 1
                    textSize = cv2.getTextSize (dominant_emotion, font, fontScale, fontThickness)[0]
                    
                    if y <= 5.0:
                        t_tl = (x, y+textSize[1]+5)
                        t_br = (x+textSize[0], y) 
                        cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2)
                        cv2.rectangle(frame, t_tl, t_br, (255, 0, 255), cv2.FILLED)
                        cv2.putText (frame, dominant_emotion, (x, y+textSize[1]), font, fontScale, (255, 255, 255), fontThickness)
                        
                    
                    else:
                        t_tl = (x, y-textSize[1]-5)
                        t_br = (x+textSize[0], y) 
                        cv2.rectangle(frame, (x, y, w, h), (255, 0, 255), 2)
                        cv2.rectangle(frame, t_tl, t_br, (255, 0, 255), cv2.FILLED)
                        cv2.putText (frame, dominant_emotion, (x, y-textSize[1]), font, fontScale, (255, 255, 255), fontThickness)
                        
                cv2.imshow ("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        video.release()
        
