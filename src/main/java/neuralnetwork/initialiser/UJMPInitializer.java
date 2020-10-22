package neuralnetwork.initialiser;

import java.util.ArrayList;
import java.util.List;
import math.linearalgebra.Matrix;
import math.linearalgebra.ujmp.UJMPMatrix;

public class UJMPInitializer extends ParameterInitializer<org.ujmp.core.Matrix> {

    public UJMPInitializer(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(weightMethod, biasMethod);
        this.wM = weightMethod;
        this.bM = biasMethod;
    }

    public List<Matrix<org.ujmp.core.Matrix>> getWeightParameters() {
        List<Matrix<org.ujmp.core.Matrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new UJMPMatrix(this.wM.initialisationValues(0, current, next)));
        }
        return weights;
    }

    public List<Matrix<org.ujmp.core.Matrix>> getBiasParameters() {
        List<Matrix<org.ujmp.core.Matrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new UJMPMatrix(this.bM.initialisationValues(0, current, 1)));
        }
        return biases;
    }

    @Override
    public List<Matrix<org.ujmp.core.Matrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    @Override
    public List<Matrix<org.ujmp.core.Matrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    protected List<Matrix<org.ujmp.core.Matrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<org.ujmp.core.Matrix>> deltaParams = new ArrayList<>();
        InitialisationMethod m = MethodConstants.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new UJMPMatrix(m.initialisationValues(0, current, next)));
        }
        return deltaParams;
    }

    @Override
    public Matrix<org.ujmp.core.Matrix> getFirstBias() {
        return new UJMPMatrix(this.bM.initialisationValues(0, this.sizes[0], 1));
    }

    @Override
    public String name() {
        return "UJMPInitializer";
    }

}
