package math.evaluation;

import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class XOREvaluationFunction implements EvaluationFunction {

	public XOREvaluationFunction() {
	}

	@Override
	public Matrix evaluatePrediction(List<NetworkInput> toEvaluate) {
		int correct = 0;
		for (NetworkInput matrices : toEvaluate) {
			Matrix fed = matrices.getData();
			Matrix corr = matrices.getLabel();

			double fedEl = fed.getElement(0, 0);
			double corrEl = corr.getElement(0, 0);

			// TODO: Might need some tweaking..
			if (Math.abs(fedEl - corrEl) < 0.05) {
				correct++;
			}
		}
		return Matrix.fromArray(new double[]{correct});
	}
}
