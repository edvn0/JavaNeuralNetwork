package math.error_functions;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class MeanSquaredCostFunction implements CostFunction {

    /**
     *
     */
    private static final long serialVersionUID = 4470711763150915089L;

    @Override
    public double calculateCostFunction(final List<NetworkInput> tData) {
        return tData.parallelStream().map((e) -> e.getData().subtract(e.getLabel()))
                .map(e -> e.multiply(e)).map(Matrix::sum).reduce(Double::sum).get() / tData.size();

    }

    @Override
    public OjAlgoMatrix applyCostFunctionGradient(final OjAlgoMatrix in, final OjAlgoMatrix correct) {
        return in.subtract(correct).multiply(2);
    }

}
