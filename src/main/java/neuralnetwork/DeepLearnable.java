package neuralnetwork;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;
import utilities.types.Pair;

public interface DeepLearnable<M> {

	void train(List<NetworkInput<M>> training, int epochs);

	void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation, int epochs, int batchSize);

	void trainWithMetrics(List<NetworkInput<M>> training, List<NetworkInput<M>> validation, int epochs, int batchSize,
			String outputPath);

	double testEvaluation(List<NetworkInput<M>> unfedTestingData, int samples);

	double testLoss(List<NetworkInput<M>> unfedTestingData);

	void display();

	void copyParameters(List<Matrix<M>> weights, List<Matrix<M>> biases);

	List<Pair<Matrix<M>, Matrix<M>>> getParameters();

	Matrix<M> predict(Matrix<M> input);
}
