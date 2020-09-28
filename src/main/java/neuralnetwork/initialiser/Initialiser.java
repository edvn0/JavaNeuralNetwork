package neuralnetwork.initialiser;

import lombok.Data;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.util.List;

public abstract class Initialiser {

    public abstract ParameterContainer createMatrices();

    public abstract OjAlgoMatrix[] clearDeltas();

    public abstract OjAlgoMatrix[] deltaParameters(OjAlgoMatrix[] biases);
}
