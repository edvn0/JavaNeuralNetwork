package math.activations;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class LinearFunction extends ActivationFunction {


    private static final long serialVersionUID = 1398037053480589797L;
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
