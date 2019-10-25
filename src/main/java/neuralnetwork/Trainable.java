package neuralnetwork;

import matrix.Matrix;

public interface Trainable {

	void train(double[] in, double[] correct);

	Matrix predict(double[] in);

}
