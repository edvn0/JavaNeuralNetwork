package neuralnetwork.initialiser;

import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.linearalgebra.ujmp.UJMPMatrix;

import java.util.ArrayList;
import java.util.List;

public class UJMPInitialiser extends ParameterInitialiser<org.ujmp.core.Matrix> {

    private int[] sizes;
    private InitialisationMethod wM, bM;

    public UJMPInitialiser(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(weightMethod, biasMethod);
        this.wM = weightMethod;
        this.bM = biasMethod;
    }

    public void init(int[] sizes) {
        this.sizes = sizes.clone();
    }

    public List<Matrix<org.ujmp.core.Matrix>> getWeightParameters() {
        List<Matrix<org.ujmp.core.Matrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new UJMPMatrix(this.wM.initialisationValues(0, current, next), current, next));
        }
        return weights;
    }

    public List<Matrix<org.ujmp.core.Matrix>> getBiasParameters() {
        List<Matrix<org.ujmp.core.Matrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new UJMPMatrix(this.bM.initialisationValues(0, current, 1), current, 1));
        }
        return biases;
    }

    public List<Matrix<org.ujmp.core.Matrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    public List<Matrix<org.ujmp.core.Matrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    protected List<Matrix<org.ujmp.core.Matrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<org.ujmp.core.Matrix>> deltaParams = new ArrayList<>();
        InitialisationMethod m = InitialisationMethod.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new UJMPMatrix(m.initialisationValues(0, current, next), current, next));
        }
        return deltaParams;
    }

}
