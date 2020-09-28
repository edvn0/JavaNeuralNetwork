package math.error_functions;

import java.util.List;
import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class CrossEntropyCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 5041727275192756048L;

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		return tData.parallelStream()
				.map(e -> e.getData().plus(1e-6).log(Ret.NEW).times(e.getLabel()).times(-1).getValueSum())
				.reduce(Double::sum).get() / tData.size();
	}

	@Override
	public Matrix applyCostFunctionGradient(final Matrix input, final Matrix correct) {
		return input.minus(correct);
	}

}
