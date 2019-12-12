package math.evaluation;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;

public interface EvaluationFunction extends Serializable {

	DenseMatrix evaluatePrediction(List<NetworkInput> toEvaluate);

	default DenseMatrix evaluateSingle(NetworkInput toEvaluate) {
		return evaluatePrediction(Collections.singletonList(toEvaluate));
	}

}
