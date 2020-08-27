package math.error_functions;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class BinaryCrossEntropyCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -5304955386755460591L;

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {

		Matrix onesData = Matrix.Factory.ones(tData.get(0).getData().getRowCount(), 1);
		Matrix onesLabel = Matrix.Factory.ones(tData.get(0).getLabel().getRowCount(), 1);

		double total = 0;
		for (NetworkInput s : tData) {

			Matrix label = s.getLabel();
			Matrix data = s.getData();

			Matrix log2Data = label.times(data.log2(Ret.NEW));
			Matrix onesMinusLabel = onesLabel.minus(label);
			Matrix onesMinusData = onesData.minus(data);
			Matrix partTwo = onesMinusLabel.times(onesMinusData.log2(Ret.NEW));

			Matrix out = log2Data.plus(partTwo);
			total += out.doubleValue() / data.getRowCount();
		}
		return (total * -1) / tData.size();
	}

	@Override
	public Matrix applyCostFunctionGradient(final Matrix in, final Matrix label) {
		return in.minus(label);
	}
}
