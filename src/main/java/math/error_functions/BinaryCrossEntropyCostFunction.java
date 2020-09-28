package math.error_functions;

import java.util.List;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.calculation.Calculation;

public class BinaryCrossEntropyCostFunction implements CostFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -5304955386755460591L;

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {

		OjAlgoMatrix onesData = OjAlgoMatrix.ones(tData.get(0).getData().rows(), 1);
		OjAlgoMatrix onesLabel = OjAlgoMatrix.ones(tData.get(0).getLabel().rows(), 1);

		double total = 0;
		for (NetworkInput s : tData) {

			OjAlgoMatrix label = s.getLabel();
			OjAlgoMatrix data = s.getData();

			OjAlgoMatrix log2Data = label.multiply(data.mapElements(e -> Math.log(e)/Math.log(2)));
			OjAlgoMatrix onesMinusLabel = onesLabel.subtract(label);
			OjAlgoMatrix onesMinusData = onesData.subtract(data);
			OjAlgoMatrix partTwo = onesMinusLabel.multiply(onesMinusData.mapElements(e -> Math.log(e)/Math.log(2)));

			OjAlgoMatrix out = log2Data.add(partTwo);
			total += out.sum() / data.rows();
		}
		return (total * -1) / tData.size();
	}

	@Override
	public OjAlgoMatrix applyCostFunctionGradient(final OjAlgoMatrix in, final OjAlgoMatrix correct) {
		return in.subtract(correct);
	}
}
