package math.activations.functional;

import org.ujmp.core.Matrix;

public interface DifferentiableFunction {

    Matrix function(Matrix m);

    Matrix derivative(Matrix m);

}
