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

		return tData.parallelStream().map(e -> e.getData().minus(e.getLabel()))
			.map(e -> e.times(e).doubleValue()).reduce(Double::sum).get() / tData.size();

	}

	@Override
	public DenseMatrix applyCostFunctionGradient(final DenseMatrix in, final DenseMatrix label) {
		return (DenseMatrix) in.minus(label).times(2);
	}

}
