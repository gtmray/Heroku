import cv2
from tensorflow.keras import models
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUs: ", len(physical_devices))
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


class VideoCamera(object):

    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.img = np.zeros((200, 400, 3), np.uint8)
        self.color = (36, 255, 12)
        self.model = models.load_model('model.h5')
        self.b = 100
        self.g = 255
        self.r = 100
        self.a = 0
        self.fontpath = "ArialUnicodeMS.ttf"
        self.font_nep = ImageFont.truetype(self.fontpath, 32)
        self.org = (50, 50)

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        
        _, img = self.video.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 1)
            crop_img = img[y:y+h, x:x+w]

        #Take cropped image for prediction if found
        if len(faces) != 0:
            img_for_pred = crop_img
            self.org = (x, y - 50)

        else:
            img_for_pred = img

        #Font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 0.9

        # Line thickness of 2 px
        thickness = 2

        #As trained in the model
        img_size = 100

        #Change the input image as per model
        img_pred = cv2.resize(img_for_pred, (img_size, img_size))
        img_pred = np.reshape(img_pred, [1, img_size, img_size, 3])

        #Prediction
        classes = self.model.predict_classes(img_pred)

        #For Nepali script
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        if classes == 1:
            #message = 'Masks laa khaate'
            message = 'कृपया मास्क लाउनु होला। '
        else:
            #message = 'Sahi ho masks laaunu parxa'
            message = 'धन्यबाद मास्क लाउनुभएकोमा। '

        cv2.putText(img, message, self.org, font,
                    fontScale, self.color, thickness, cv2.LINE_AA)

        #Properties set for Nepali script
        draw.text(self.org, message, font=self.font_nep, fill=(self.b, self.g, self.r, self.a))
        img = np.array(img_pil)

        # Display
        #cv2.imshow('img', img)

        # cv2.putText(img, message, self.org, font,
        #             fontScale, self.color, thickness, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', img)

        return jpeg.tobytes()
