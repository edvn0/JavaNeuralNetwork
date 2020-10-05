package math.activations;

import math.linearalgebra.Matrix;

public class DoNothingFunction<M> extends ActivationFunction<M> {


    public DoNothingFunction() {
    }

    @Override
    public String getName() {
        return "DoNothing";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m;
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m;
    }

    @Override
    public Matrix<M> derivativeOnInput(Matrix<M> input, Matrix<M> out) {
        return out.hadamard(derivative(input));
    }

}
