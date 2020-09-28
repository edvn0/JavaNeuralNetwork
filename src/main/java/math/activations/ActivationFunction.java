package math.activations;

import math.activations.functional.DifferentiableFunction;
import org.ujmp.core.Matrix;

import java.io.Serializable;

public abstract class ActivationFunction implements DifferentiableFunction, Serializable {

    public ActivationFunction() {

    }

    public Matrix derivativeOnInput(Matrix input, Matrix out) {
        return out.times(derivative(input));
    }

    public abstract String getName();
}
