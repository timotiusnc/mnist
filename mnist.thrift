typedef list<list<list<double>>> Image

struct Prediction {
	1: double digit,
	2: double probability
}

service MNIST {
	list<Prediction> predict(1: Image image)
	void retrain(1: Image image, 2: double digit)
}
