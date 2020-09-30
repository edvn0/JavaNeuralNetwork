package neuralnetwork.inputs;

import math.linearalgebra.Matrix;

public class LabeledNetworkInput<T, M> extends NetworkInput<M> {

    /**
     *
     */
    private static final long serialVersionUID = -2389046210309718893L;
    private final T inputLabel;

    public LabeledNetworkInput(T inputLabel, Matrix<M> data, Matrix<M> label) {
        super(data, label);
        this.inputLabel = inputLabel;
    }

    public T getInputLabel() {
        return inputLabel;
    }

}
