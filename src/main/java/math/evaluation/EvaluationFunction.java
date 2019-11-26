package math.evaluation;

import java.io.Serializable;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public interface EvaluationFunction extends Serializable {

	Matrix evaluatePrediction(List<NetworkInput> toEvaluate);

}
