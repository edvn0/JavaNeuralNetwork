package math.errors;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class CrossEntropyCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 5041727275192756048L;

	public CrossEntropyCostFunction() {

	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		for (NetworkInput input : tData) {
			sum +=
				(input.getData().plus(1e-6).log(Ret.NEW)
					.times(input.getLabel())
					.getValueSum()) * -1;
		}

		return (sum) / tData.size();
	}

	@Override
	public DenseMatrix applyCostFunctionGradient(final DenseMatrix input,
		final DenseMatrix label) {
		return (DenseMatrix) input.minus(label);
	}

}
