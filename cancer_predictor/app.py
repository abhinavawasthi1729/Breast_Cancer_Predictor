from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

model=pickle.load(open('model1.pkl','rb'))

@app.route('/result', methods=['POST'])
def home():
    radius_mean=request.form['radius_mean']
    texture_mean=request.form['texture_mean']
    smoothness_mean=request.form['smoothness_mean']
    compactness_mean=request.form['compactness_mean']
    concavity_mean=request.form['concavity_mean']
    symmetry_mean=request.form['symmetry_mean']
    fractal_dimension_mean=request.form['fractal_dimension_mean']

    arr=np.array([[radius_mean,texture_mean,smoothness_mean, compactness_mean,concavity_mean,symmetry_mean,fractal_dimension_mean]])
    pred=model.predict(arr)
    return render_template('index.html',data=pred)

if(__name__=="__main__"):
    app.run(debug=True)