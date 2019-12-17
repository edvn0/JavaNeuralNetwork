package math.errors;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;

public class MeanSquaredCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 4470711763150915089L;

	public MeanSquaredCostFunction() {
	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		for (NetworkInput networkInput : tData) {
			DenseMatrix label = networkInput.getLabel();
			DenseMatrix data = networkInput.getData();
			DenseMatrix inner = (DenseMatrix) label.minus(data);
			sum += inner.times(inner).doubleValue();
		}

		sum /= tData.size();

		return sum;
	}

	@Override
	public DenseMatrix applyCostFunctionGradient(final DenseMatrix in, final DenseMatrix label) {
		return (DenseMatrix) in.minus(label).times(2);
	}

}