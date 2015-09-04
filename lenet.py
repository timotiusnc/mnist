import caffe

MODEL_PATH = "lenet.caffemodel"
PROTOBUF_PATH = "lenet.prototxt"

class LeNet:
	def __init__(self):
		caffe.set_mode_cpu()

		self.net = caffe.Net(PROTOBUF_PATH, MODEL_PATH, caffe.TEST)

		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data', (2, 0, 1))

	def predict(self, image):
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

		probs = self.net.forward(end='prob')['prob'].flatten()
		argmaxs = self.net.forward(start='prob')['argmax'].flatten()

		return [(argmax, probs[argmax]) for argmax in argmaxs]
