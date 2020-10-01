package math.activations;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class DoNothingFunction extends ActivationFunction {

    private static final long serialVersionUID = -1697257154018408060L;

    public DoNothingFunction() {
    }

    @Override
    public String getName() {
        return "DoNothing";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return m;
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return m;
    }

}
