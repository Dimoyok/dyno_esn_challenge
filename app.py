from flask import Flask, render_template
from flask_sock import Sock
import time
import numpy as np

app = Flask(__name__)
sock = Sock(app)
control_value = 0

W_res = np.load("model2.npy")
W_out, W, W_in, W_c = W_res[:, :4].T, W_res[:, 4:304], W_res[:, 304:308], W_res[:, 308:]

@app.route('/')
def index():
    return render_template('index.html')

def lorenz_step(y, rho = 28, dt = 0.002):
    sigma = 10
    beta = 8/3.

    r = y.copy()
    r[0] = y[0] + dt * sigma * (y[1] - y[0])
    r[1] = y[1] + dt * (y[0] * (rho - y[2]) - y[1])
    r[2] = y[2] + dt * (y[0] * y[1] - beta * y[2])

    return r

def esn_step(y, s, c):
    # print(W_in.shape, W.shape, W_c.shape, y.shape, s.shape, c.shape)
    s = np.tanh(W_in @ y + W @ s + W_c @ c)
    y = W_out @ s
    return y, s

@sock.route('/to_server')
def to_server(sock):
    global control_value
    while True:
        data = sock.receive()
        control_value = np.array(list(map(float, data.split())))
        # print(control_value)

@sock.route('/from_server')
def from_server(sock):
    global control_value

    control_value = np.array([0.2, -0.2])
    y = np.array([1, -0.38557625, -0.47640575,  0.0317102 ])
    s = np.zeros((300,))
    while True:
        time.sleep(1 / 30.)
        y, s = esn_step(y, s, control_value)
        y[0] = 1
        sock.send(" ".join(map(str, y[1:])))




if __name__ == '__main__':
    app.run(debug=True)