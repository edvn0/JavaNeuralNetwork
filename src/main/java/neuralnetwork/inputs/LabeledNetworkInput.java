package neuralnetwork.inputs;

import org.ujmp.core.Matrix;

public class LabeledNetworkInput<T> extends NetworkInput {

    private final T inputLabel;

    public LabeledNetworkInput(T inputLabel, Matrix data, Matrix label) {
        super(data, label);
        this.inputLabel = inputLabel;
    }

    public T getInputLabel() {
        return inputLabel;
    }



}
