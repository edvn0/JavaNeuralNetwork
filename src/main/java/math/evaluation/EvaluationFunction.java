package math.evaluation;

import neuralnetwork.inputs.NetworkInput;
import utilities.serialise.NetworkSerialisable;

import java.util.Collections;
import java.util.List;

/**
 * Evaluate a set of test data against some strategy, like thresholds or
 * ArgMaxing.
 */
public interface EvaluationFunction<M> extends NetworkSerialisable<String, Double> {

    public void init(double... in);

    /**
     * Evaluates a list of {@link NetworkInput}, either training or test data.
     *
     * @param toEvaluate the data set to evaluate and compare to its label.
     * @return a double representing the percentage correct score.
     */
    double evaluatePrediction(List<NetworkInput<M>> toEvaluate);

    /**
     * Evaluate a single training example
     *
     * @param toEvaluate a single training / test / validation input.
     * @return a double representing the percentage correct score.
     */
    default double evaluateSingle(NetworkInput<M> toEvaluate) {
        return evaluatePrediction(Collections.singletonList(toEvaluate));
    }

    String name();

}
