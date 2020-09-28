package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class LinearFunction extends ActivationFunction {

    private final double value;

    public LinearFunction(double value) {
        this.value = value;
    }

    @Override
    public String getName() {
        return "Linear";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return m.mapElements((e) -> e * value);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return m.mapElements((e) -> value);
    }
}
