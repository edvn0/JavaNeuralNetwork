package math.evaluation;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;

public interface EvaluationFunction extends Serializable {

	/**
	 * Evaluates a list of {@link NetworkInput}, either training or test data.
	 *
	 * @param toEvaluate the data set to evaluate and compare to its label.
	 *
	 * @return a matrix representing the correct score.
	 */
	DenseMatrix evaluatePrediction(List<NetworkInput> toEvaluate);

	/**
	 * Evaluate a single training example
	 *
	 * @param toEvaluate a single training / test / validation input.
	 *
	 * @return a matrix representing the correct score.
	 */
	default DenseMatrix evaluateSingle(NetworkInput toEvaluate) {
		return evaluatePrediction(Collections.singletonList(toEvaluate));
	}

}
