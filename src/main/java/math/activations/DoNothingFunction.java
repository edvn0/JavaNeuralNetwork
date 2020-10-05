package math.activations;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class DoNothingFunction<M> extends ActivationFunction<M> {

    private static final long serialVersionUID = -1697257154018408060L;

    public DoNothingFunction() {
    }

    @Override
    public String getName() {
        return "DoNothing";
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m;
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m;
    }

    @Override
    public Matrix<M> derivativeOnInput(Matrix<M> input, Matrix<M> out) {
        return out.hadamard(derivative(input));
    }

}
