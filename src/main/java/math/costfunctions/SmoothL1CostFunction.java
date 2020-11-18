package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class SmoothL1CostFunction<M> implements CostFunction<M> {

    private double l1 = 1;

    public void setL1(double l1) {
        this.l1 = l1;
    }

    @Override
    public double calculateCostFunction(List<NetworkInput<M>> tData) {

        if (tData.size() == 1) {
            return calcuateSingle(tData.get(0));
        }

        double huber = 0d;
        for (var d : tData) {
            huber += calcuateSingle(d);
        }
        return huber / tData.size();
    }

    @Override
    public Matrix<M> applyCostFunctionGradient(Matrix<M> in, Matrix<M> correct) {
        Matrix<M> diff = correct.subtract(in);
        return diff.mapElements(e -> {
            if (Math.abs(e) < l1) {
                return Math.abs(e);
            } else {
                return Math.signum(e);
            }
        });
    }

    @Override
    public double calcuateSingle(NetworkInput<M> data) {
        Matrix<M> diff = data.getLabel().subtract(data.getData());
        return diff.mapElements(e -> {
            if (Math.abs(e) < l1) {
                return e * e / 2;
            } else {
                return l1 * Math.abs(e) - (l1 * l1) / 2;
            }
        }).sum();
    }

    @Override
    public String name() {
        return "Huber Loss";
    }

}
