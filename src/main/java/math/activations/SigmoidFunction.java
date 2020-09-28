package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

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
    public Matrix function(Matrix m) {
        return MatrixUtilities.map(m.clone(), this::sigmoid);
    }

    @Override
    public Matrix derivative(Matrix m) {
        return MatrixUtilities.map(m.clone(), this::sigmoidDerivative);
    }
}
