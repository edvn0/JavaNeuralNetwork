package neuralnetwork.inputs;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.io.Serializable;

/**
 * A wrapper to contain a input to a Neural Network, with both the label and the
 * data.
 */
public class NetworkInput implements Serializable {

    /**
     *
     */
    private static final long serialVersionUID = -8743845031383184256L;
    private final OjAlgoMatrix data;
    private final OjAlgoMatrix label;

    public NetworkInput(OjAlgoMatrix data, OjAlgoMatrix label) {
        this.data = data;
        this.label = label;
    }

    public OjAlgoMatrix getData() {
        return data;
    }

    public OjAlgoMatrix getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return "NetworkInput{" + "data=" + data + ", label=" + label + '}';
    }
}
