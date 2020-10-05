package math.evaluation;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class ThresholdEvaluationFunction<M> implements EvaluationFunction<M> {

    /**
     *
     */
    private static final long serialVersionUID = 2502293428392120484L;
    private static final String NAME = "Threshold Evaluation";
    private final double threshHold;

    public ThresholdEvaluationFunction(double a) {
        this.threshHold = a;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double evaluatePrediction(List<NetworkInput<M>> toEvaluate) {
        int correct = 0;
        for (NetworkInput<M> matrices : toEvaluate) {
            Matrix<M> fed = matrices.getData();
            Matrix<M> corr = matrices.getLabel();

            double fedEl = fed.sum();
            double corrEl = corr.sum();

            if (Math.abs(fedEl - corrEl) < this.threshHold) {
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
