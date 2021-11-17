from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import train

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/train")
def training():
    train.train()
    return render_template('index.html')

if __name__ == '__main__':
    app.run()