package math.activations.functional;

import math.linearalgebra.Matrix;

public interface DifferentiableFunction<M> {

	Matrix<M> function(Matrix<M> m);

	Matrix<M> derivative(Matrix<M> m);

}
