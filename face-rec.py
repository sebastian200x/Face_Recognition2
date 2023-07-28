from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('capture.html')

@app.route('/verify')
def about():
    return render_template('verify.html')

if __name__ == '__main__':
    app.run(debug=True)