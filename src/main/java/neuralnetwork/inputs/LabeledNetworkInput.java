package neuralnetwork.inputs;

import math.linearalgebra.Matrix;

public class LabeledNetworkInput<T, M> extends NetworkInput<M> {

    private final T inputLabel;

    public LabeledNetworkInput(T inputLabel, Matrix<M> data, Matrix<M> label) {
        super(data, label);
        this.inputLabel = inputLabel;
    }

    public T getInputLabel() {
        return inputLabel;
    }


}
