from flask import Flask, render_template, request
import tensorflow as tf
app = Flask(__name__)
def triangle_area(a, b, c):
    s = (a + b + c) / 2.0
    area = tf.sqrt(s * (s - a) * (s - b) * (s - c))
    return area.numpy()
def circle_area(r):
    r = tf.cast(r, dtype=tf.float32)
    pi = tf.constant(22/7, dtype=tf.float32)
    area = tf.multiply(pi, r, r)
    return area.numpy()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/calculate', methods=['POST'])
def calculate():
    a = float(request.form['side_a'])
    b = float(request.form['side_b'])
    c = float(request.form['side_c'])
    area = triangle_area(a, b, c)
    return render_template("result.html", area=area)
@app.route('/calculatecirclearea', methods=['POST'])
def calculatecirclearea():
    r = float(request.form['side_a'])
    area=circle_area(r)
    return render_template("arearesult.html", area=area)

if __name__ == '__main__':
    app.run(debug=True)