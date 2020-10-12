package neuralnetwork;

import java.util.List;
import neuralnetwork.inputs.NetworkInput;

public interface DeepLearnable<M> {

	void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation, int epochs,
		int batchSize);

	void trainWithMetrics(List<NetworkInput<M>> training, List<NetworkInput<M>> validation,
		int epochs,
		int batchSize, String outputPath);

	double testEvaluation(List<NetworkInput<M>> unfedTestingData, int samples);

	double testLoss(List<NetworkInput<M>> unfedTestingData);

	void display();
}
