from flask import Flask, request, jsonify, render_template

#Initialize the flask App
app = Flask(__name__)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/proposal')
def proposal():
    return render_template('proposal.html')

if __name__ == "__main__":
    app.run(debug=True)