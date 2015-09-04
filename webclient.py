import sys
sys.path.append("./gen-py")

import base64
import urllib

from flask import Flask, render_template, request
from caffe.io import load_image, resize_image
from skimage.color import rgb2gray

from mnist import MNIST
from mnist.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


asu = Flask(__name__)


@asu.route('/')
def home():
    return render_template('index.html')


@asu.route('/classify', methods=['POST'])
def classify():
    file = request.files['file']

    if file:
	file.save('upload')
        orig_img = load_image('upload', color=False)
        img = preprocess_image(orig_img)
        ret = send_image(img)
        return render_template('result.html', result=ret, img=base64.b64encode(open('upload', 'r').read()))
    return 'Upload failed!'


def preprocess_image(img):
    prep = img
    if prep.shape == 3 and prep.shape[2] > 1:
        prep = rgb2gray(prep)
    if prep.shape != (28, 28, 1):
        prep = resize_image(prep, (28, 28, 1))
    return prep


def send_image(img):
    try:
        # Make socket
        transport = TSocket.TSocket('localhost', 9090)
        # Buffering is critical. Raw sockets are very slow
        transport = TTransport.TBufferedTransport(transport)
        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        # Create a client to use the protocol encoder
        client = MNIST.Client(protocol)

        transport.open()
        # ret = [(p.digit, p.probability) for p in client.predict(img)]
        ret = client.predict(img)
        transport.close()

        return ret
    except Thrift.TException, tx:
        print '%s' % (tx.message)


if __name__ == "__main__":
    asu.run(debug=True)
