package neuralnetwork.initialiser;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.util.ArrayList;
import java.util.List;

public class ParameterInitialiser {

    private int[] sizes;
    private InitialisationMethod wM, bM;

    public ParameterInitialiser(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        this.wM = weightMethod;
        this.bM = biasMethod;
    }

    public void init(int[] sizes) {
        this.sizes = sizes.clone();
    }

    public List<OjAlgoMatrix> getWeightParameters() {
        List<OjAlgoMatrix> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new OjAlgoMatrix(this.wM.initialisationValues(0, current, next), current, next));
        }
        return weights;
    }

    public List<OjAlgoMatrix> getBiasParameters() {
        List<OjAlgoMatrix> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new OjAlgoMatrix(this.bM.initialisationValues(0, current, 1), current, 1));
        }
        return biases;
    }

    public List<OjAlgoMatrix> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    public List<OjAlgoMatrix> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    private List<OjAlgoMatrix> getDeltaParameters(boolean isBias) {
        List<OjAlgoMatrix> deltaParams = new ArrayList<>();
        InitialisationMethod m = InitialisationMethod.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new OjAlgoMatrix(m.initialisationValues(0, current, next), current, next));
        }
        return deltaParams;
    }

}
