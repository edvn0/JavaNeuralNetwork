package math.evaluation;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class ThreshHoldEvaluationFunction implements EvaluationFunction {

	private double threshHold;

	/**
	 *
	 */
	private static final long serialVersionUID = 2502293428392120484L;

	public ThreshHoldEvaluationFunction(double a) {
		this.threshHold = a;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DenseMatrix evaluatePrediction(List<NetworkInput> toEvaluate) {
		int correct = 0;
		for (NetworkInput matrices : toEvaluate) {
			DenseMatrix fed = matrices.getData();
			DenseMatrix corr = matrices.getLabel();

			double fedEl = fed.getValueSum();
			double corrEl = corr.getValueSum();

			if (Math.abs(fedEl - corrEl) < this.threshHold) {
				correct++;
			}
		}
		return Matrix.Factory.importFromArray(new double[]{correct});
	}
}
