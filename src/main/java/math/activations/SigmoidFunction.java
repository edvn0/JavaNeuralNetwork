package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class SigmoidFunction extends ActivationFunction {

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
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return m.mapElements(this::sigmoid);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return m.mapElements(this::sigmoidDerivative);
    }
}
