package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class MeanSquaredCostFunction<M> implements CostFunction<M> {

    private static final String NAME = "Mean Squared Error";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {

        if (tData.size() == 1) {
            return calcuateSingle(tData.get(0));
        }

        double mse = 0;
        for (var d : tData) {
            mse += calcuateSingle(d);
        }

        return mse / tData.size();
    }

    @Override
    public double calcuateSingle(NetworkInput<M> data) {
        var ni = data;
        var diff = ni.getData().subtract(ni.getLabel());
        return diff.hadamard(diff).sum();
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
