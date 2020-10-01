package neuralnetwork.inputs;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class LabeledNetworkInput<T> extends NetworkInput {

    /**
     *
     */
    private static final long serialVersionUID = -2389046210309718893L;
    private final T inputLabel;

    public LabeledNetworkInput(T inputLabel, OjAlgoMatrix data, OjAlgoMatrix label) {
        super(data, label);
        this.inputLabel = inputLabel;
    }

    public T getInputLabel() {
        return inputLabel;
    }

}
