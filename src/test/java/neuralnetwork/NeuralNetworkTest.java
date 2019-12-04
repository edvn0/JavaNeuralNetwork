package neuralnetwork;

import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {

	NeuralNetwork nn;

	@Before
	public void setUp() {
		ActivationFunction[] functions = new ActivationFunction[4];
		functions[0] = new ReluFunction();
		functions[1] = new ReluFunction();
		functions[2] = new ReluFunction();
		functions[3] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		nn = new NeuralNetwork(0.035, functions, function, eval,
			new int[]{3, 10, 10, 10});
	}

	@Test
	public void getNow() {
		System.out.println(NeuralNetwork.getNow());
		String basePath = "/home/edwin98carlsson/";
		String use = basePath.endsWith("/") ? basePath : basePath + "/";

		String loss = use + "LossToEpochPlot";
		String correct = use + "CorrectToEpochPlot";

		String now = NeuralNetwork.getNow();

		String nowLoss = loss + "_" + now + ".jpg";
		String nowCorr = correct + "_" + now + ".jpg";

		System.out.println(nowLoss);
		System.out.println(nowCorr);
	}

	@Test
	public void xavierInitialization() {
		System.out.println(nn.xavierInitialization(3));
		System.out.println(nn.xavierInitialization(100));
		System.out.println(nn.xavierInitialization(100));
	}
}