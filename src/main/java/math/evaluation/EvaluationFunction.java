package math.evaluation;

import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public interface EvaluationFunction {

	Matrix evaluatePrediction(List<NetworkInput> toEvaluate);

}
