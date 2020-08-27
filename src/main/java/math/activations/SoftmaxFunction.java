package math.activations;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import org.ujmp.core.Matrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import utilities.MatrixUtilities;

public class SoftmaxFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -5298468440584699205L;

	/**
	 * Takes as input a vector of size NX1 and returns a SoftMax Vector of that
	 * input.
	 *
	 * @param input input vector.
	 *
	 * @return softmax vector.
	 */
	private Matrix softMax(Matrix input) {
		if (input.getColumnCount() != 1) {
			throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
		}

		Matrix max = this.max(input);
		Matrix z = input.minus(max);
		double sum = DoubleStream.of(MatrixUtilities.toArray(z)).map(Math::exp).sum();

		return MatrixUtilities.map(z, (e) -> Math.exp(e) / sum);
	}

	private Matrix max(Matrix input) {
		double[] out = new double[(int) input.getRowCount()];
		int max = input.indexOfMax(Ret.NEW, 0).intValue();
		Arrays.fill(out, max);
		return Matrix.Factory.importFromArray(out).transpose();
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		return this.softMax(input);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		return null;
	}

	@Override
	public Matrix derivativeOnInput(final Matrix input, final Matrix out) {
		double xOut = input.times(out).getValueSum();
		Matrix derive = out.minus(xOut);
		return input.times(derive);
	}

	@Override
	public String getName() {
		return "SOFTMAX";
	}
}
