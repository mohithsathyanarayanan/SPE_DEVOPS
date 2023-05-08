from flask import Flask,render_template, request, redirect, send_from_directory
import cv2
import numpy as np
import imutils
from os import listdir
import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import unittest
import logging
logging.basicConfig(filename='logfile.log',format='%(asctime)s %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)
#logging.info('This is an info message')
def get_image(i):       
	path = "./sample_test_images/"+str(i)+".jpg"
	return path      

class TestModel(unittest.TestCase):
    def test1(self):        
        result = detect(get_image(1))
        print("Prediction of Img ",1,":- ",int(result<0.9))
        logger.info("Prediction of Img 1:- %d ",result)
        self.assertEqual(1, int(result<0.9))
        
      
    def test2(self):        
        result = detect(get_image(2))
        print("Prediction of Img ",2,":- ",result)
        logger.info("Prediction of Img 2:- %d ",result)
        self.assertEqual(0, int(result<0.9))
        
      
    def test3(self):        
        result= detect(get_image(3))
        print("Prediction of Img ",3,":- ",result)
        logger.info("Prediction of Img 3:- %d ",result)
        self.assertEqual(1, int(result<0.9))
    

      
    def test4(self):        
        result = detect(get_image(4))
        print("Prediction of Img ",4,":- ",result)
        logger.info("Prediction of Img 4:- %d ",result)
        self.assertEqual(1, int(result<0.9))
        
      
    def test5(self):        
        result = detect(get_image(5))
        print("Prediction of Img ",5,":- ",result)
        logger.info("Prediction of Img 5:- %d ",result)
        self.assertEqual(1, int(result<0.9))
        
              
    def test6(self):        
        result = detect(get_image(6))
        print("Prediction of Img ",6,":- ",result)
        logger.info("Prediction of Img 6:- %d ",result)
        self.assertEqual(0, int(result<0.9))
        
      
    def test7(self):        
        result = detect(get_image(7))
        print("Prediction of Img ",7,":- ",result)
        logger.info("Prediction of Img 7:- %d ",result)
        self.assertEqual(1, int(result<0.9))

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
    return new_image


def detect(img):
    model=load_model('cnn-parameters-improvement-23-0.91.model')
    image = cv2.imread(img)
    image = preprocess(image)
    image = cv2.resize(image, dsize=(240,240), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    X=[]
    X.append(image)
    X = np.array(X)
    y=model.predict(X)
    return round(y[0][0],2)


if __name__ =="__main__":
    unittest.main()
