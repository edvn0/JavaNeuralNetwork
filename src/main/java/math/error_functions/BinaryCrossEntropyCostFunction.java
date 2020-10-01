package math.error_functions;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;

import java.util.List;

public class BinaryCrossEntropyCostFunction implements CostFunction {

    private static final double log2 = Math.log(2);
    private static final long serialVersionUID = -5304955386755460591L;
    private static final String NAME = "Binary Cross Entropy";

    @Override
    public double calculateCostFunction(final List<NetworkInput> tData) {

        NetworkInput firstElement = tData.get(0);
        OjAlgoMatrix firstData = firstElement.getData();

        OjAlgoMatrix onesData = firstData.ones(firstData.rows(), 1);
        OjAlgoMatrix onesLabel = firstData.ones(firstElement.getLabel().rows(), 1);

        double total = 0;
        for (NetworkInput s : tData) {

            OjAlgoMatrix label = s.getLabel();
            OjAlgoMatrix data = s.getData();

            OjAlgoMatrix log2Data = label.multiply(data.mapElements(e -> Math.log(e) / log2));
            OjAlgoMatrix onesMinusLabel = onesLabel.subtract(label);
            OjAlgoMatrix onesMinusData = onesData.subtract(data);
            OjAlgoMatrix partTwo = onesMinusLabel.multiply(onesMinusData.mapElements(e -> Math.log(e) / log2));

            OjAlgoMatrix out = log2Data.add(partTwo);
            total += out.sum() / data.rows();
        }
        return (-total) / tData.size();
    }

    @Override
    public OjAlgoMatrix applyCostFunctionGradient(final OjAlgoMatrix in, final OjAlgoMatrix correct) {
        return in.subtract(correct);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
