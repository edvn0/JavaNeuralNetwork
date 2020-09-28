package math.error_functions;

import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class CrossEntropyCostFunction implements CostFunction {

    /**
     *
     */
    private static final long serialVersionUID = 5041727275192756048L;

    @Override
    public double calculateCostFunction(final List<NetworkInput> tData) {
        // TODO: IMPLEMENT
        return 0;
    }

    @Override
    public OjAlgoMatrix applyCostFunctionGradient(final OjAlgoMatrix input, final OjAlgoMatrix correct) {
        return input.subtract(correct);
    }

}
