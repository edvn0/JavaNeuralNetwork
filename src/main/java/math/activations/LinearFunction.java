package math.activations;

import math.linearalgebra.Matrix;

public class LinearFunction<M> extends ActivationFunction<M> {

    private final double value;

    public LinearFunction(double value) {
        this.value = value;
    }

    @Override
    public String getName() {
        return "Linear";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m.mapElements((e) -> e * value);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m.mapElements((e) -> value);
    }
}
