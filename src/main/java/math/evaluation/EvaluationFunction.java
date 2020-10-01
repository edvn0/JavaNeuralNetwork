package math.evaluation;

import neuralnetwork.inputs.NetworkInput;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * Evaluate a set of test data against some strategy, like thresholds or
 * ArgMaxing.
 */
public interface EvaluationFunction extends Serializable {

    /**
     * Evaluates a list of {@link NetworkInput}, either training or test data.
     *
     * @param toEvaluate the data set to evaluate and compare to its label.
     * @return a double representing the percentage correct score.
     */
    double evaluatePrediction(List<NetworkInput> toEvaluate);

    /**
     * Evaluate a single training example
     *
     * @param toEvaluate a single training / test / validation input.
     * @return a double representing the percentage correct score.
     */
    default double evaluateSingle(NetworkInput toEvaluate) {
        return evaluatePrediction(Collections.singletonList(toEvaluate));
    }

    String toString();

}
