package math.activations;

import math.activations.functional.DifferentiableFunction;
import math.linearalgebra.Matrix;

public abstract class ActivationFunction<M> implements DifferentiableFunction<M> {

    public ActivationFunction() {

    }

    public abstract void setValues(double in);

    public Matrix<M> derivativeOnInput(Matrix<M> input, Matrix<M> out) {
        return out.hadamard(derivative(input));
    }

    public abstract String getName();
}
