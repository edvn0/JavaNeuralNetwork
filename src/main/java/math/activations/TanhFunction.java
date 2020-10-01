package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class TanhFunction extends ActivationFunction {

    private static final long serialVersionUID = 9220582961801243285L;

    private double tanh(double a) {
        return Math.tanh(a);
    }

    private double tanhDerivative(double a) {
        return 1 - (a * a);
    }

    @Override
    public String getName() {
        return "Tanh";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return m.mapElements(this::tanh);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return m.mapElements(this::tanhDerivative);
    }
}
