package math.activations.functional;

import math.linearalgebra.Matrix;

public interface DifferentiableFunction<M> {

    M function(M m);

    M derivative(M m);

}
