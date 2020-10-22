package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class MeanSquaredCostFunction<M> implements CostFunction<M> {

    private static final String NAME = "Mean Squared Error";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {
        double mse = 0;
        for (var d : tData) {
            var diff = d.getData().subtract(d.getLabel());
            var squared = diff.hadamard(diff);
            mse += squared.sum();
        }

        return mse / tData.size();
    }

    @Override
    public Matrix<M> applyCostFunctionGradient(final Matrix<M> in, final Matrix<M> correct) {
        return in.subtract(correct).multiply(2);
    }

    @Override
    public String name() {
        return NAME;
    }
}
