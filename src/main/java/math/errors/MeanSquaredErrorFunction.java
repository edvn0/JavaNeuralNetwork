package math.errors;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class MeanSquaredErrorFunction implements ErrorFunction {

	public MeanSquaredErrorFunction() {
	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		for (NetworkInput networkInput : tData) {
			DenseMatrix inner = (DenseMatrix) networkInput.getLabel().minus(networkInput.getData());
			inner = MatrixUtilities.map(inner, e -> e * e / 2);
			sum += inner.getValueSum();
		}

		sum /= tData.size();

		return sum;
	}

	@Override
	public DenseMatrix applyErrorFunction(final DenseMatrix input, final DenseMatrix target) {
		return (DenseMatrix) input.minus(target);
	}

	@Override
	public DenseMatrix applyErrorFunctionGradient(final DenseMatrix in, final DenseMatrix label) {
		return (DenseMatrix) in.minus(label);
	}

}
