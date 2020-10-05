package math.error_functions;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class CrossEntropyCostFunction<M> implements CostFunction<M> {

    private static final long serialVersionUID = 5041727275192756048L;
    private static final String NAME = "Cross Entropy";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {
        double size = (double) tData.size();

        return -tData.parallelStream().map(e -> e.getData().mapElements(Math::log).hadamard(e.getLabel()))
                .mapToDouble(Matrix::sum).sum() / size;

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
