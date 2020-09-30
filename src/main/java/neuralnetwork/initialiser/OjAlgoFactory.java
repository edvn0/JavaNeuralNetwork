package neuralnetwork.initialiser;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class OjAlgoFactory extends ParameterFactory<OjAlgoMatrix> {
    public OjAlgoFactory(int[] sizes, InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(sizes, weightMethod, biasMethod);
    }

    @Override
    public List<Matrix<OjAlgoMatrix>> getWeightParameters() {
        List<Matrix<OjAlgoMatrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new OjAlgoMatrix(this.wM.initialisationValues(0, current, next), current, next));
        }
        return weights;
    }

    @Override
    public List<Matrix<OjAlgoMatrix>> getBiasParameters() {
        List<Matrix<OjAlgoMatrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new OjAlgoMatrix(this.bM.initialisationValues(0, current, 1), current, 1));
        }
        return biases;
    }

    @Override
    public List<Matrix<OjAlgoMatrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    @Override
    public List<Matrix<OjAlgoMatrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    @NotNull
    private List<Matrix<OjAlgoMatrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<OjAlgoMatrix>> deltaParams = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new OjAlgoMatrix(this.wM.initialisationValues(0, current, next), current, next));
        }
        return deltaParams;
    }

}
