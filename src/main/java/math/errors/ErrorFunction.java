package math.errors;

import java.io.Serializable;
import matrix.Matrix;

public interface ErrorFunction extends Serializable {

	Matrix applyErrorFunction(Matrix input, Matrix target);

}
