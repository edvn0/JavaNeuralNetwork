package math.activations;

import math.activations.functional.DifferentiableFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.io.Serializable;

public abstract class ActivationFunction implements DifferentiableFunction, Serializable {

    public ActivationFunction() {

    }

    public OjAlgoMatrix derivativeOnInput(OjAlgoMatrix input, OjAlgoMatrix out) {
        return out.multiply(derivative(input));
    }

    public abstract String getName();
}
