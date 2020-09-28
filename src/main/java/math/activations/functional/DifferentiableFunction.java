package math.activations.functional;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public interface DifferentiableFunction {

    OjAlgoMatrix function(OjAlgoMatrix m);

    OjAlgoMatrix derivative(OjAlgoMatrix m);

}
