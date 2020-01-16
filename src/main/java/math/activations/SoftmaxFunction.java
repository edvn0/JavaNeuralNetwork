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

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		return this.softMax(input);
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		return null;
	}

	@Override
	public DenseMatrix derivativeOnInput(final DenseMatrix input, final DenseMatrix out) {
		double xOut = input.times(out).getValueSum();
		DenseMatrix derive = (DenseMatrix) out.minus(xOut);
		return (DenseMatrix) input.times(derive);
	}

	public SoftmaxFunction() {

	}

	@Override
	public String getName() {
		return "SOFTMAX";
	}
}
