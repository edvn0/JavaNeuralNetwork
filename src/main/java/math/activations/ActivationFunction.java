package math.activations;

import math.activations.functional.DifferentiableFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.io.Serializable;

import lombok.extern.slf4j.Slf4j;

public abstract class ActivationFunction implements DifferentiableFunction<OjAlgoMatrix>, Serializable {

    /**
     *
     */
    private static final long serialVersionUID = 5640850328132384749L;

    public ActivationFunction() {

    }

    public OjAlgoMatrix derivativeOnInput(OjAlgoMatrix input, OjAlgoMatrix out) {
        return out.hadamard(derivative(input));
    }

    public abstract String getName();
}
