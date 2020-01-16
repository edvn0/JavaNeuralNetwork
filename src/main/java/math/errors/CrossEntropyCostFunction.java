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
		return tData
			.parallelStream()
			.map(e ->
				e.getData()
					.plus(1e-6)
					.log(Ret.NEW)
					.times(
						e.getLabel()
					)
					.times(-1)
					.getValueSum())
			.reduce(Double::sum)
			.get()
			/
			tData.size();
	}

	@Override
	public DenseMatrix applyCostFunctionGradient(final DenseMatrix input,
		final DenseMatrix label) {
		return (DenseMatrix) input.minus(label);
	}

}
