package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class LeakyReluFunction extends ReluFunction {

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
    public Matrix derivative(Matrix in) {
        return MatrixUtilities.map(in, (e) -> e > 0 ? e : alpha);
    }

    @Override
    public Matrix function(Matrix in) {
        return MatrixUtilities.map(in, (e) -> e > 0 ? 1 : alpha);
    }
}
