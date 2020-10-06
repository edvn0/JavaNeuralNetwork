package math.costfunctions;

import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class BinaryCrossEntropyCostFunction<M> implements CostFunction<M> {

    private static final double log2 = Math.log(2);
    private static final String NAME = "Binary Cross Entropy";

    @Override
    public double calculateCostFunction(final List<NetworkInput<M>> tData) {

        NetworkInput<M> firstElement = tData.get(0);
        Matrix<M> firstData = firstElement.getData();

        Matrix<M> onesData = firstData.ones(firstData.rows(), 1);
        Matrix<M> onesLabel = firstData.ones(firstElement.getLabel().rows(), 1);

        double total = 0;
        for (var s : tData) {

            Matrix<M> label = s.getLabel();
            Matrix<M> data = s.getData();

            Matrix<M> log2Data = label.multiply(data.mapElements(e -> Math.log(e) / log2));
            Matrix<M> onesMinusLabel = onesLabel.subtract(label);
            Matrix<M> onesMinusData = onesData.subtract(data);
            Matrix<M> partTwo = onesMinusLabel.multiply(onesMinusData.mapElements(e -> Math.log(e) / log2));

            Matrix<M> out = log2Data.add(partTwo);
            total += out.sum() / data.rows();
        }
        return (-total) / tData.size();
    }

    @Override
    public Matrix<M> applyCostFunctionGradient(final Matrix<M> in, final Matrix<M> correct) {
        return in.subtract(correct);
    }

    @Override
    public String name() {
        return NAME;
    }
}
