import numpy as np
import cv2
from tensorflow.keras.models import load_model

###parameters###

width = 640
height = 480
threshold = 0.65

#threshold means minimum probability to classify

# Load the image file instead of using video input
image_path = "regenerated_braille_image.png"
imgOriginal = cv2.imread(image_path)

# Check if the image was loaded successfully
if imgOriginal is None:
    print("Error: Can't open/read image file. Check file path/integrity.")
    exit()

# Load the saved pre-trained model
model = load_model('model.h5')

#this is the code for processing
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

img = np.asarray(imgOriginal)
img=cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processsed Image",img)
img = img.reshape(1, 32, 32, 1)

#prediction
#prediction
predictions = model.predict(img)
classIndex = np.argmax(predictions)

labelDictionary = {0: '0', 1: 'අ', 2: 'ඉ', 3: 'ඊ', 4: 'උ', 5: 'එ', 6: 'ඒ', 7: 'ඔ', 8: 'ක', 9: 'ක්', 10: 'කා',
                   11: 'කැ', 12: 'කෑ', 13: 'කි', 14: 'කී', 15: 'කු', 16: 'කූ', 17: 'කෙ', 18: 'කේ', 19: 'කො',
                   20: 'කෝ', 21: 'ඛ', 22: 'ග', 23: 'ගි', 24: 'ගී', 25: 'ගු', 26: 'ගූ', 27: 'ඝ', 28: 'ඟ', 29: 'ච',
                   30: 'ඡ', 31: 'ජ', 32: 'ජ්', 33: 'ජි', 34: 'ජී', 35: 'ඣ', 36: 'ඤ', 37: 'ඥ', 38: 'ට', 39: 'ඨ',
                   40: 'ඩ',
                   41: 'ඪ', 42: 'ණ', 43: 'ඬ', 44: 'ත', 45: 'ත්', 46: 'ථ', 47: 'ථි', 48: 'ථී', 49: 'ද', 50: 'දු',
                   51: 'දූ', 52: 'ධ', 53: 'න', 54: 'ඳ', 55: 'ප', 56: 'පු'}

predictions = model.predict(img)
classIndex = np.argmax(predictions)
predictedLetter=labelDictionary.get(classIndex)
probabilityValue = np.amax(predictions)
print(predictedLetter, probabilityValue,classIndex)

if probabilityValue > threshold:
    cv2.putText(imgOriginal, str(predictedLetter) + "   " + str(probabilityValue),
                (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 1)

cv2.imshow("Testing Window", imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()

