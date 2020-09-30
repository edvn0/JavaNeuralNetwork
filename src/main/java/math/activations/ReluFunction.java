package math.activations;

import math.linearalgebra.Matrix;

public class ReluFunction<M> extends ActivationFunction<M> {
    private static final long serialVersionUID = 1430626027686849916L;

    public ReluFunction() {
        super();
    }

    @Override
    public String getName() {
        return "ReLU";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m.mapElements((e) -> e > 0 ? e : 0);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m.mapElements((e) -> e > 0 ? 1d : 0);
    }
}
