from flask import Flask, render_template, redirect, url_for, request
import shutil
import os
import pandas as pd
import numpy as np
import librosa
from scipy.io import wavfile
import tensorflow as tf
from pydub import AudioSegment
from python_speech_features import mfcc
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
MODEL_PATH = 'model/species.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded")

def envelope(y,rate, threshold):
  mask=[]
  y = pd.Series(y).apply(np.abs)
  y_mean=y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
  for mean in y_mean:
    if mean>threshold:
      mask.append(True)
    else:
      mask.append(False)
  return mask

@app.route('/')
def hello_world():
    shutil.rmtree('uploads/')
    os.mkdir('uploads')
    shutil.rmtree('clean/')
    os.mkdir('clean')
    return render_template("start.html")

@app.route('/', methods=['POST'])
def predictor():
    if request.method == 'POST':
        audio = request.files['chooseFile']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, "uploads", secure_filename(audio.filename)
        )
        print(file_path)
        audio.save(file_path)
        file_name = secure_filename(audio.filename).split('.')
        src = file_path
        dst = 'uploads/'+file_name[0] + ".wav"
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
        os.remove(file_path)
        file_path = os.path.join(
            basepath, "uploads", file_name[0] + ".wav"
        )
        signal, rate = librosa.load( file_path, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/' + file_name[0] + ".wav", rate=rate, data=signal[mask])
        y_pred=prediction('clean/' + file_name[0] + ".wav")
        flat_pred = []
        for i in y_pred:
            for j in i:
                flat_pred.extend(j)
        y_pred = np.argmax(y_pred)
        confidence=flat_pred[y_pred]*100
        confidence=round(confidence,2)
        print("confidence: ",confidence)
        y_pred=(y_pred)%3
        print(y_pred)
        print("Done")
        if confidence>96:
            return render_template("index.html", result=y_pred, confidence=confidence)
        else:
             return render_template("404.html")

def prediction(audioloc):
  y_pred=[]
  print("Extracting audio features")
  rate,wav = wavfile.read(audioloc)
  for i in range(0,wav.shape[0]-1600, 1600):
    sample = wav[i:i+1600]
    x = mfcc(sample, rate, numcep=13, nfilt=26, nfft=512)
    x = (x- (-100.49625436527774))/(107.55423014451932 - (-100.49625436527774))
    x = x.reshape(1, x.shape[0], x.shape[1],1 )
    y_hat = model.predict(x)
    y_pred.append(y_hat)
  return y_pred

@app.route('/about', methods=['GET', 'POST'])
def open_about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
