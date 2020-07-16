# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:29:36 2020

@author: Laptop
"""
import pickle
import flask
import os

print("Load model . . .")
mapGebruikers = "C:\\Users\\"
mapGebruiker = os.getlogin()
mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
os.chdir(hoofdmap)

app = flask.Flask(__name__)
port=int(os.getenv("PORT",9099))

model = pickle.load(open(r'data/optimodel_SVM_SL_KOD.pck', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    features = flask.request.get_json(force=True)['features']
    prediction = model.predict([features])[0,0]
#   prediction_proba = model.predict_proba([features])[0,0]
    response = {'prediction': prediction}
        
    return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
    
# json input gewenst is:
# {
#         "features": [feature1]
# } 

# json response terug is:
# {
#         "features": [feature1]
# } 


