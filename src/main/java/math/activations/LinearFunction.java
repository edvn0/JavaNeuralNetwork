package math.activations;

import math.linearalgebra.Matrix;

public class LinearFunction<M> extends ActivationFunction<M> {

    private double value;

    public LinearFunction(double value) {
        this.value = value;
    }

    public LinearFunction() {
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

    @Override
    public void setValues(double in) {
        this.value = in;
    }
}
