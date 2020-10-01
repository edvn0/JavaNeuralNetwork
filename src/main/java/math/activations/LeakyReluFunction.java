package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class LeakyReluFunction extends ReluFunction {

    private static final long serialVersionUID = -301187113858930644L;
    private final double alpha;

    public LeakyReluFunction(double alpha) {
        super();
        this.alpha = alpha;
    }

    @Override
    public String getName() {
        return "LeakyReLU";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix in) {
        return in.mapElements((e) -> e > 0 ? e : alpha * e);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix in) {
        return in.mapElements((e) -> e > 0 ? 1 : alpha);
    }

}
