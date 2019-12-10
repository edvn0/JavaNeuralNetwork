package math.activations;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import utilities.MatrixUtilities;

public class SoftmaxFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -5298468440584699205L;

	/**
	 * Takes as input a vector of size NX1 and returns a SoftMax Vector of that input.
	 *
	 * @param input input vector.
	 *
	 * @return softmax vector.
	 */
	private DenseMatrix softMax(DenseMatrix input) {
		if (input.getColumnCount() != 1) {
			throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
		}

		DenseMatrix max = this.max(input);
		DenseMatrix z = (DenseMatrix) input.minus(max);
		double sum = DoubleStream.of(MatrixUtilities.toArray(z)).map(Math::exp).sum();

		return MatrixUtilities.map(z, (e) -> Math.exp(e) / sum);
	}

	private DenseMatrix max(DenseMatrix input) {
		double[] out = new double[(int) input.getRowCount()];
		int max = input.indexOfMax(Ret.NEW, 0).intValue();
		Arrays.fill(out, max);
		return (DenseMatrix) Matrix.Factory.importFromArray(out).transpose();
	}

	private double max(double[][] data) {
		double max = data[0][0];
		for (int i = 1; i < data.length; i++) {
			if (data[i][0] > max) {
				max = data[i][0];
			}
		}
		return max;
	}

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		return this.softMax(input);
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		return MatrixUtilities.map(input, (e) -> (e * (1d - e)));
	}

	public SoftmaxFunction() {

	}

	@Override
	public String getName() {
		return "SOFTMAX";
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("SoftmaxFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
