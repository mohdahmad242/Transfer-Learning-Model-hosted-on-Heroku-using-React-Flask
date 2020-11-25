from ml_model.predict import pred

import os
from flask import Flask, render_template, request, make_response, jsonify, send_file

app = Flask(__name__)

# Set up the main route
@app.route('/predict', methods=['POST'])
def home():
	print("Action Initiated")
	review = request.json['review']
	prediction = pred(review)
	print(prediction)
	return prediction

if __name__ == '__main__':
    app.run()
