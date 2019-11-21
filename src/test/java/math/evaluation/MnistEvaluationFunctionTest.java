package math.evaluation;

import java.util.ArrayList;
import java.util.List;
import matrix.Matrix;
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
		Matrix[] c = new Matrix[]{output, correct};
		Matrix[] c2 = new Matrix[]{output, incorrect};
		List<Matrix[]> preds = new ArrayList<>();
		preds.add(c);
		preds.add(c2);
		int corrects = (int) f.evaluatePrediction(preds).getElement(0, 0);
		Assert.assertEquals(corrects, 1);
		Assert.assertEquals((int) output.getElement(3, 0), (int) correct.getElement(3, 0));
		Assert.assertNotEquals((int) output.getElement(3, 0), (int) incorrect.getElement(3, 0));
	}
}