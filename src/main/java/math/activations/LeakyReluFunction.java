package math.activations;

import math.linearalgebra.Matrix;

public class LeakyReluFunction<M> extends ReluFunction<M> {

    private final double alpha;

    public LeakyReluFunction(double alpha) {
        super();
        this.alpha = alpha;
    }

    @Override
    public String getName() {
        return "LeakyReLU";
    }

    @Override
    public Matrix<M> function(Matrix<M> in) {
        return in.mapElements((e) -> e > 0 ? e : alpha * e);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> in) {
        return in.mapElements((e) -> e > 0 ? 1 : alpha);
    }

}
