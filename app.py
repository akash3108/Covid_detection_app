from werkzeug.utils import secure_filename
import Prediction as pred
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def covid_prediction():
    file = request.files['file'].read()
    npimg = np.fromstring(file, np.uint8)
    
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = pred.get_prediction(img)
    result = "Covid Negative" if result==0 else "Covid Positive"
    return jsonify({"prediction":result})


if __name__ == '__main__':
    app.run()
