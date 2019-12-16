package math.errors;

import java.io.Serializable;
import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;

public interface CostFunction extends Serializable {

	double calculateCostFunction(List<NetworkInput> tData);

	DenseMatrix applyCostFunctionGradient(DenseMatrix in, DenseMatrix label);

}
