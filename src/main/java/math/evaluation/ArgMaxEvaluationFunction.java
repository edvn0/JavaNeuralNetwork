package math.evaluation;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

/**
 * Evaluation function which returns a correct value iff the argmax of the
 * predicted data is the label
 */
public class ArgMaxEvaluationFunction implements EvaluationFunction {

    /**
     *
     */
    private static final long serialVersionUID = 3730260463010183881L;
    private static final String NAME = "Argmax Evaluation";

    /**
     * {@inheritDoc}
     */
    @Override
    public double evaluatePrediction(List<NetworkInput> toEvaluate) {
        int correct = 0;
        for (NetworkInput networkInput : toEvaluate) {
            OjAlgoMatrix data = networkInput.getData();
            int correctLabels = networkInput.getLabel().argMax();
            int val = data.argMax();

            if (correctLabels == val) {
                correct++;
            }

        }
        return (double) correct / toEvaluate.size();
    }

    @Override
    public String toString() {
        return NAME;
    }
}
