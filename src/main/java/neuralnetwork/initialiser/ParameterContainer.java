package neuralnetwork.initialiser;

import lombok.Data;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

@Data
public class ParameterContainer {
    private OjAlgoMatrix[] parameters;
    private OjAlgoMatrix[] deltaParameters;
}
