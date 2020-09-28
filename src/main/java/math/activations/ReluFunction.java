package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class ReluFunction extends ActivationFunction {

    public ReluFunction() {
        super();
    }

    @Override
    public String getName() {
        return "ReLU";
    }

    @Override
    public Matrix function(Matrix m) {
        return MatrixUtilities.map(m.clone(), (e) -> e > 0 ? e : 0);
    }

    @Override
    public Matrix derivative(Matrix m) {
        return MatrixUtilities.map(m.clone(), (e) -> e > 0 ? 1d : 0);
    }
}
