package math.error_functions;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class MeanSquaredCostFunction implements CostFunction {

    private static final long serialVersionUID = 4470711763150915089L;
    private static final String NAME = "Mean Squared Error";

    @Override
    public double calculateCostFunction(final List<NetworkInput> tData) {
        return tData.parallelStream().map((e) -> e.getData().subtract(e.getLabel())).mapToDouble(e -> e.square()).sum()
                / tData.size();

    }

    @Override
    public OjAlgoMatrix applyCostFunctionGradient(final OjAlgoMatrix in, final OjAlgoMatrix correct) {
        return in.subtract(correct).multiply(2);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
