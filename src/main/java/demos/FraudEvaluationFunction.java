package demos;

import java.util.List;
import math.evaluation.EvaluationFunction;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class FraudEvaluationFunction implements EvaluationFunction {

	@Override
	public DenseMatrix evaluatePrediction(final List<NetworkInput> list) {
		int correct = 0;
		for (NetworkInput p : list) {
			if (Math.abs(p.getData().doubleValue() - p.getLabel().doubleValue()) < 10e-5) {
				correct++;
			}
		}
		return Matrix.Factory.importFromArray(new double[][]{{correct}});
	}
}
