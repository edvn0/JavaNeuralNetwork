package math.activations;

import math.linearalgebra.Matrix;

public class SoftmaxFunction<M> extends ActivationFunction<M> {

    /**
     * Takes as input a vector of size NX1 and returns a SoftMax Vector of that
     * input.
     *
     * @param input input vector.
     * @return softmax vector.
     */
    private Matrix<M> softMax(Matrix<M> input) {
        if (input.cols() != 1) {
            throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
        }
        Matrix<M> max = input.maxVector();
        Matrix<M> z = input.subtract(max);
        double sum = z.mapElements(Math::exp).sum();

        return z.mapElements((e) -> Math.exp(e) / sum);
    }

    @Override
    public Matrix<M> derivativeOnInput(final Matrix<M> input, final Matrix<M> out) {
        double xOut = input.hadamard(out).sum();
        Matrix<M> derive = out.subtract(xOut);
        return input.hadamard(derive);
    }

    @Override
    public String getName() {
        return "Softmax";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return this.softMax(m);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return null;
    }

    @Override
    public void setValues(double in) {

    }
}
