package math.errors;

import java.util.Arrays;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class CrossEntropyErrorFunction implements ErrorFunction {

	public CrossEntropyErrorFunction() {

	}

	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;
		//tData.forEach(
		//	el -> sum[0] += el.getLabel()
		//		.dotProduct(el.getData().map((e) -> Math.log(e) / Math.log(2))));

		double[] stable = new double[tData.get(0).getData().getRows()];
		Arrays.fill(stable, 1e-8);

		Matrix stability = Matrix.fromArray(stable);

		for (NetworkInput input : tData) {
			Matrix label = input.getLabel();
			Matrix data = input.getData();
			data = data.add(stability);
			Matrix logged = data.map((e) -> Math.log(e) / Math.log(2));
			double dot = label.dotProduct(logged);
			sum += dot;
		}

		return (sum * -1) / tData.size();
	}

	@Override
	public Matrix applyErrorFunction(final Matrix in, final Matrix correct) {
		return null;
	}

	@Override
	public Matrix applyErrorFunctionGradient(final Matrix target, final Matrix applied) {
		return target.subtract(applied);
	}

}
