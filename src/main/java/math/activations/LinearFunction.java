package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class LinearFunction extends ActivationFunction {

    private final double value;

    public LinearFunction(double value) {
        this.value = value;
    }

    @Override
    public String getName() {
        return "Linear";
    }

    @Override
    public Matrix function(Matrix m) {
        return MatrixUtilities.map(m.clone(), (e) -> e * value);
    }

    @Override
    public Matrix derivative(Matrix m) {
        return MatrixUtilities.map(m.clone(), (e) -> value);
    }
}
