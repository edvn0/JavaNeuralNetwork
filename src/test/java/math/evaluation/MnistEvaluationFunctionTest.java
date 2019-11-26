package math.evaluation;

import java.util.ArrayList;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;
import org.junit.Assert;
import org.junit.Test;

public class MnistEvaluationFunctionTest {

	private final MnistEvaluationFunction f = new MnistEvaluationFunction();

	@Test
	public void evaluatePrediction() {
		Matrix output = Matrix
			.fromArray(new double[]{0.001, 0.003, 0.13, 1, 0.15, 0.01, 0.0001, 0.022, 0.18});
		Matrix correct = Matrix.fromArray(new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0});
		Matrix incorrect = Matrix.fromArray(new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0});
		NetworkInput c = new NetworkInput(output, correct);
		NetworkInput c2 = new NetworkInput(output, incorrect);
		List<NetworkInput> preds = new ArrayList<>();
		preds.add(c);
		preds.add(c2);
		int corrects = (int) f.evaluatePrediction(preds).getElement(0, 0);
		Assert.assertEquals(corrects, 1);
		Assert.assertEquals((int) output.getElement(3, 0), (int) correct.getElement(3, 0));
		Assert.assertNotEquals((int) output.getElement(3, 0), (int) incorrect.getElement(3, 0));
	}
}