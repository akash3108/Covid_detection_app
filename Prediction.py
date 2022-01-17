import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import cv2
import pickle
from scipy import stats
import OsteoArthNet as OAN
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def get_prediction(img):
    # Resize image & Change it into array
    img = cv2.resize(img, (224,224))
    img = image.img_to_array(img)
    img = img.reshape(1,224,224,3)

    # Load Osteoarthnet
    model_osteoarthnet = OAN.BuildModel()
    model = model_osteoarthnet.OsteoArthNet()
    model.load_weights('./Model_Checkpoints/SARS-COVID.ckpt')
    osteoarthet = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Load Classifier and Scalar
    with open('./classifier_scalar.pkl', 'rb') as f:
        models, scalar = pickle.load(f)

    # Extract Features
    extracted_feature = osteoarthet.predict(img)
    extracted_feature = extracted_feature.reshape(1,2048)
    
    # Scale extracted feature
    extracted_feature = scalar.transform(extracted_feature)

    # Make prediction
    predictions = []
    for model in models:
        pred = model.predict(extracted_feature)
        predictions.append(pred[0])
        
    output = stats.mode(predictions)[0][0]

    return output