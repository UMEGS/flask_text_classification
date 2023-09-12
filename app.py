from flask import Flask, render_template, request
from transformers import pipeline
import time

app = Flask(__name__)

classifier = pipeline("sentiment-analysis")


@app.route('/')
def index():
    return render_template('index.html')


def make_response(res, time_taken):
    pre = {}

    if res[0]['label'] == 'NEGATIVE':
        pre['POSITIVE'] = 1 - round(res[0]['score'], 2)
        pre['NEGATIVE'] = round(res[0]['score'], 2)

    if res[0]['label'] == 'POSITIVE':
        pre['POSITIVE'] = round(res[0]['score'], 2)
        pre['NEGATIVE'] = 1 - round(res[0]['score'], 2)

    pre['time_taken'] = round(time_taken, 2)

    return pre


@app.route('/predict', methods=['POST'])
def classifications():
    text = request.form['text']

    start_time = time.time()

    res = classifier(text)
    time_taken = (time.time() - start_time) * 1000

    res = make_response(res, time_taken)

    return res


if __name__ == '__main__':
    app.run()
