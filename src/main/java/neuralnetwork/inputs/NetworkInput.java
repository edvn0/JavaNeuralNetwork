package neuralnetwork.inputs;

import org.ujmp.core.Matrix;

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
    private final Matrix data;
    private final Matrix label;

    public NetworkInput(Matrix data, Matrix label) {
        this.data = data;
        this.label = label;
    }

    public Matrix getData() {
        return data;
    }

    public Matrix getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return "NetworkInput{" + "data=" + data + ", label=" + label + '}';
    }
}
