package math.activations;

import math.linearalgebra.Matrix;

public class TanhFunction<M> extends ActivationFunction<M> {


    private double tanh(double a) {
        return Math.tanh(a);
    }

    private double tanhDerivative(double a) {
        return 1 - (a * a);
    }

    @Override
    public String getName() {
        return "Tanh";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m.mapElements(this::tanh);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m.mapElements(this::tanhDerivative);
    }
}
