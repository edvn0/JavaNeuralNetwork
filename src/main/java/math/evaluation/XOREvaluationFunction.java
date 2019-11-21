package math.evaluation;

import java.util.List;
import matrix.Matrix;

public class XOREvaluationFunction implements EvaluationFunction {

	@Override
	public Matrix evaluatePrediction(List<Matrix[]> toEvaluate) {
		int correct = 0;
		for (Matrix[] matrices : toEvaluate) {
			Matrix fed = matrices[0];
			Matrix corr = matrices[1];

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
