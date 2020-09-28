package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class TanhFunction extends ActivationFunction {

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
    public Matrix function(Matrix m) {
        return MatrixUtilities.map(m.clone(), this::tanh);
    }

    @Override
    public Matrix derivative(Matrix m) {
        return MatrixUtilities.map(m.clone(), this::tanhDerivative);
    }
}
