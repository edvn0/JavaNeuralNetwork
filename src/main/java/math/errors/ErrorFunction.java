package math.errors;

import java.io.Serializable;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public interface ErrorFunction extends Serializable {

	double calculateCostFunction(List<NetworkInput> tData);

	Matrix applyErrorFunction(Matrix in, Matrix correct);

	Matrix applyErrorFunctionGradient(Matrix in, Matrix applied);

}
