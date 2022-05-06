from flask import Flask, render_template, request
from flask_sock import Sock
import time
import numpy as np

app = Flask(__name__)
sock = Sock(app)
data = {}

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

def esn_step(_d):
    _d["states"] = np.tanh(W_in @ _d["outputs"] + 
        W @ _d["states"] + W_c @ _d["controls"])
    _d["outputs"] = W_out @ _d["states"]
    _d["outputs"][0] = 1

def update_data(addr):
    # print(data.keys())
    if(addr in data): data[addr]["timer"] = time.time()
    for a in list(data.keys()):
        if(time.time() - data[a]["timer"] > 0.5):
            del data[a] 
    if(addr in data): return

    data[addr] = {}
    data[addr]["controls"] = np.array([0.2, -0.2])
    data[addr]["states"] = np.zeros((300,))
    data[addr]["outputs"] = np.array([1, -0.38557625, -0.47640575,  0.0317102 ])
    data[addr]["timer"] = time.time()


@sock.route('/to_server')
def to_server(sock):
    global control_value
    while True:
        update_data(request.remote_addr)

        recieved = sock.receive()
        _d = data[request.remote_addr]
        _d["controls"] = np.array(list(map(float, recieved.split())))

@sock.route('/from_server')
def from_server(sock):
    global control_value

    while True:
        update_data(request.remote_addr)

        time.sleep(1 / 30.)
        _d = data[request.remote_addr]
        esn_step(_d)
        sock.send(" ".join(map(str, _d["outputs"][1:])))



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)