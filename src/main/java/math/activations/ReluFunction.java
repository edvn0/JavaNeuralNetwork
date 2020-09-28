package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class ReluFunction extends ActivationFunction {

    public ReluFunction() {
        super();
    }

    @Override
    public String getName() {
        return "ReLU";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return m.mapElements((e) -> e > 0 ? e : 0);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return m.mapElements((e) -> e > 0 ? 1d : 0);
    }
}
