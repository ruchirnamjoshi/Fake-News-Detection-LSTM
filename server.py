from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)



@app.route('/predict_fake_news', methods=['POST'])
def predict_fake_news(news):



    response = util.predict_fake_news(news)
    return response
@app.route('/', methods=['POST'])
def home():
    return render_template('app.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = util.predict_fake_news(message)
        print(pred)
        return render_template('app1.html',prediction=pred)
    else:
        return render_template('app.html', prediction="Something went wrong")


if __name__ == '__main__':
        app.run(debug=True)
