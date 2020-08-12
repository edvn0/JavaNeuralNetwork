package math.error_functions;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class BinaryCrossEntropyCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -5304955386755460591L;

	public BinaryCrossEntropyCostFunction() {
	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {

		DenseMatrix onesData = DenseMatrix.Factory.ones(tData.get(0).getData().getRowCount(), 1);
		DenseMatrix onesLabel = DenseMatrix.Factory.ones(tData.get(0).getLabel().getRowCount(), 1);

		double total = 0;
		for (NetworkInput s : tData) {

			DenseMatrix label = s.getLabel();
			DenseMatrix data = s.getData();

			DenseMatrix log2Data = (DenseMatrix) label.times(data.log2(Ret.NEW));
			DenseMatrix onesMinusLabel = (DenseMatrix) onesLabel.minus(label);
			DenseMatrix onesMinusData = (DenseMatrix) onesData.minus(data);
			DenseMatrix partTwo = (DenseMatrix) onesMinusLabel.times(onesMinusData.log2(Ret.NEW));

			DenseMatrix out = (DenseMatrix) log2Data.plus(partTwo);
			total += out.doubleValue() / data.getRowCount();
		}
		return (total * -1) / tData.size();
	}

	@Override
	public DenseMatrix applyCostFunctionGradient(final DenseMatrix in, final DenseMatrix label) {
		return (DenseMatrix) in.minus(label);
	}
}
