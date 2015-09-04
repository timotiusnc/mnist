#!/usr/bin/env python

import sys
sys.path.append("./gen-py")

import numpy as np

from lenet import LeNet

from mnist import MNIST
from mnist.ttypes import *

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class MNISTServer:
	def __init__(self):
		self.lenet = LeNet()

	def predict(self, image):
		preprocessed_image = preprocess(image)
		pmf = self.lenet.predict(preprocess_image)
		return [Prediction(digit=digit, probability=probability) for digit, probability in pmf]

	def preprocess(self, image):
		preprocessed_image = np.array(image, dtype='float32')
		return preprocessed_image

def main():
	processor = MNIST.Processor(MNISTServer())
	transport = TSocket.TServerSocket(port=9090)
	transport_factory = TTransport.TBufferedTransportFactory()
	protocol_factory = TBinaryProtocol.TBinaryProtocolFactory()

	server = TServer.TSimpleServer(processor, transport, transport_factory, protocol_factory)
	server.serve()

if __name__ == "__main__":
	main()
