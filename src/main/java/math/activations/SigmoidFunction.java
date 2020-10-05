package math.activations;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class SigmoidFunction<M> extends ActivationFunction<M> {

    private static final long serialVersionUID = -5780307498502440160L;

    @Override
    public String getName() {
        return "Sigmoid";
    }

    private double sigmoid(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    private double sigmoidDerivative(double input) {
        return input * (1 - input);
    }

    @Override
    public Matrix<M> function(Matrix<M> m) {
        return m.mapElements(this::sigmoid);
    }

    @Override
    public Matrix<M> derivative(Matrix<M> m) {
        return m.mapElements(this::sigmoidDerivative);
    }
}
