package math.activations;

import math.linearalgebra.Matrix;

public class LeakyReluFunction<M> extends ReluFunction<M> {

    private static final long serialVersionUID = -301187113858930644L;
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
