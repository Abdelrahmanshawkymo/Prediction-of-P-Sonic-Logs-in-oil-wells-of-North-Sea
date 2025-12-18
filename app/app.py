#importing required libraries
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for, redirect, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  


model = joblib.load('model/Predicton_of_sonic_logs.pkl')
column_transformer = joblib.load('model/column_transform.joblib')

# home page
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

# login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # check for username and password
        if username == 'Abdelrahman' and password == 'admin123':
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

# logout page
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# home page after login
@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    try:
        df = pd.read_csv(file)
        
        # check the required features
        required_features = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI']
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            return render_template('index.html', error=f'Missing columns: {", ".join(missing_cols)}')
        
        # data processing
        if 'DT' not in df.columns:
            df['DT'] = 0.0
        
        if 'DEPTH' not in df.columns:
            df['DEPTH'] = np.arange(len(df))
        
        if 'WELL' not in df.columns:
            df['WELL'] = np.arange(len(df))
        else:
            df['WELL'] = df['WELL'].astype(float)
        
        # covert numeric data
        predict_features = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI']
        df[predict_features] = df[predict_features].apply(pd.to_numeric, errors='coerce').fillna(0.1)
        df['RT'] = df['RT'].apply(lambda x: np.log(x) if x > 0 else np.log(0.1))
        
        # prediction
        X_for_model = df[predict_features]
        predicted = model.predict(X_for_model).reshape(-1, 1)
        df['Predicted_DT'] = predicted
        
        # disply the plot of predicted dt
        plt.figure(figsize=(5, 15))
        plt.plot(df['Predicted_DT'], df['DEPTH'], color='red')
        plt.gca().invert_yaxis()
        plt.xlabel("DT (µs/m)")
        plt.ylabel("Depth (m)")
        plt.title("Predicted DT")
        plt.grid()
        plt.tight_layout()
        
        # download the plot as png in the static folder
        plot_filename = secure_filename(f"prediction_{session['username']}.png")
        plot_path = os.path.join("static", plot_filename)
        plt.savefig(plot_path,bbox_inches='tight')
        plt.close()
        
        
        return render_template("result.html", image_url=url_for('static', filename=plot_filename),download_url=url_for('static', filename=plot_filename))
    
    
    except Exception as e:
        return render_template('index.html', error=f'Error processing file: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)