package math.activations;

import math.activations.functional.DifferentiableFunction;
import math.linearalgebra.Matrix;

import java.io.Serializable;

public abstract class ActivationFunction<M> implements DifferentiableFunction<M>, Serializable {

    public ActivationFunction() {

    }

    public Matrix<M> derivativeOnInput(Matrix<M> input, Matrix<M> out) {
        return out.multiply(derivative(input));
    }

    public abstract String getName();
}
