package math.activations;

import math.activations.functional.DifferentiableFunction;
import math.linearalgebra.Matrix;

import java.io.Serializable;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public abstract class ActivationFunction<M> implements DifferentiableFunction<M>, Serializable {

    /**
     *
     */
    private static final long serialVersionUID = 5640850328132384749L;

    public ActivationFunction() {

    }

    public Matrix<M> derivativeOnInput(Matrix<M> input, Matrix<M> out) {
        Matrix<M> ret = out.multiply(derivative(input));
        return ret;
    }

    public abstract String getName();
}
