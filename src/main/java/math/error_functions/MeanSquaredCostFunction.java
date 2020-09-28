package math.error_functions;

import java.util.List;
import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.Matrix;

public class MeanSquaredCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 4470711763150915089L;

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		return tData.parallelStream().map((NetworkInput e) -> e.getData().minus(e.getLabel()))
				.map(e -> e.times(e).doubleValue()).reduce(Double::sum).get() / tData.size();

	}

	@Override
	public Matrix applyCostFunctionGradient(final Matrix in, final Matrix correct) {
		return in.minus(correct).times(2);
	}

}
