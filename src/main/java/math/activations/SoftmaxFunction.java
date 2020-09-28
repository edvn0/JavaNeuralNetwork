package math.activations;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import static utilities.MatrixUtilities.map;

public class SoftmaxFunction extends ActivationFunction {

    /**
     *
     */
    private static final long serialVersionUID = -5298468440584699205L;

    /**
     * Takes as input a vector of size NX1 and returns a SoftMax Vector of that
     * input.
     *
     * @param input input vector.
     * @return softmax vector.
     */
    private Matrix softMax(Matrix input) {
        if (input.getColumnCount() != 1) {
            throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
        }

        Matrix max = this.max(input);
        Matrix z = input.minus(max);
        double sum = map(z.clone(), Math::exp).getValueSum();

        return map(z.clone(), (e) -> Math.exp(e) / sum);
    }

    private Matrix max(Matrix input) {
        double max = input.max(Ret.NEW, 0).doubleValue();
        return Matrix.Factory.zeros(input.getRowCount(), 1).plus(max);
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

    @Override
    public Matrix function(Matrix m) {
        return this.softMax(m.clone());
    }

    @Override
    public Matrix derivative(Matrix m) {
        return null;
    }
}
