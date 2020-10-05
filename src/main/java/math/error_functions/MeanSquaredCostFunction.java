package math.error_functions;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class MeanSquaredCostFunction<M> implements CostFunction<M> {

    private static final long serialVersionUID = 4470711763150915089L;
    private static final String NAME = "Mean Squared Error";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {
        return tData.parallelStream().map((e) -> e.getData().subtract(e.getLabel())).mapToDouble(e -> e.norm()).sum()
                / tData.size();

    }

    @Override
    public Matrix<M> applyCostFunctionGradient(final Matrix<M> in, final Matrix<M> correct) {
        return in.subtract(correct).multiply(2);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
