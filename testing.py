import numpy as np
import pickle
import cv2
from keras.models import load_model
import os

###parameterrs###

width = 640
height = 480
threshold = 0.65
#threshold means minimum probability to classify


#this is the code for creatinng the camera objrct
capture = cv2.VideoCapture(0)
capture.set(3,width)
capture.set(4,height)

#here im loading the saved pretrained model
pickle_in = open('model.h5')
model = pickle.load(pickle_in)
#model = tensorflow.load_model('model.h5')


# Load the saved model
#model = load_model('model.h5')

#thid is the code for processing
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = capture.read()
    img = np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1, 32, 32, 1)
    #prediction
    classIndex = int(model.predict_classes(img))

    labelDictionary = {0: '0', 1: 'අ', 2: 'ඉ', 3: 'ඊ', 4: 'උ', 5: 'එ', 6: 'ඒ', 7: 'ඔ', 8: 'ක', 9: 'ක්', 10: 'කා',
                       11: 'කැ', 12: 'කෑ', 13: 'කි', 14: 'කී', 15: 'කු', 16: 'කූ', 17: 'කෙ', 18: 'කේ', 19: 'කො',
                       20: 'කෝ', 21: 'ඛ', 22: 'ග', 23: 'ගි', 24: 'ගී', 25: 'ගු', 26: 'ගූ', 27: 'ඝ', 28: 'ඟ', 29: 'ච',
                       30: 'ඡ', 31: 'ජ', 32: 'ජ්', 33: 'ජි', 34: 'ජී', 35: 'ඣ', 36: 'ඤ', 37: 'ඥ', 38: 'ට', 39: 'ඨ',
                       40: 'ඩ',
                       41: 'ඪ', 42: 'ණ', 43: 'ඬ', 44: 'ත', 45: 'ත්', 46: 'ථ', 47: 'ථි', 48: 'ථී', 49: 'ද', 50: 'දු',
                       51: 'දූ', 52: 'ධ', 53: 'න', 54: 'ඳ', 55: 'ප', 56: 'පු', 57: 'පූ', 58: 'ඵ', 59: 'බ', 60: 'භ',
                       61: 'ම',
                       62: 'ම්', 63: 'මි', 64: 'මී', 65: 'ඹ', 66: 'ය', 67: 'ර', 68: 'ල', 69: 'ව', 70: 'ව්', 71: 'වි',
                       72: 'වී', 73: 'වු', 74: 'වූ', 75: 'ශ', 76: 'ෂ', 77: 'ස', 78: 'හ', 79: 'ළ', 80: 'ළු', 81: 'ෆ',
                       82: 'ා'}

    predictions = model.predict(img)
    predictedLetter=labelDictionary.get(classIndex)
    probabilityValue = np.amax(predictions)
    print(predictedLetter, probabilityValue)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(predictedLetter) + "   " + str(probabilityValue),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Testing Window", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

