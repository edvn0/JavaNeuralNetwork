package math.error_functions;

import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class CrossEntropyCostFunction<M> implements CostFunction<M> {

    /**
     *
     */
    private static final long serialVersionUID = 5041727275192756048L;
    private static final String NAME = "Cross Entropy";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {
        // TODO: IMPLEMENT
        return 0;
    }

    @Override
    public Matrix<M> applyCostFunctionGradient(final Matrix<M> input, final Matrix<M> correct) {
        return input.subtract(correct);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
