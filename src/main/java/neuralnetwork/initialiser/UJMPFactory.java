package neuralnetwork.initialiser;

import math.linearalgebra.Matrix;
import math.linearalgebra.ujmp.UJMPMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

import static neuralnetwork.initialiser.InitialisationMethod.ZERO;

public class UJMPFactory extends ParameterFactory<UJMPMatrix> {
    public UJMPFactory(int[] sizes, InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(sizes, weightMethod, biasMethod);
    }

    @Override
    public List<Matrix<UJMPMatrix>> getWeightParameters() {

        List<Matrix<UJMPMatrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new UJMPMatrix(this.wM.initialisationValues(0, current, next), current, next));
        }
        return weights;
    }

    @Override
    public List<Matrix<UJMPMatrix>> getBiasParameters() {
        List<Matrix<UJMPMatrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new UJMPMatrix(this.bM.initialisationValues(0, current, 1), current, 1));
        }
        return biases;
    }

    @Override
    public List<Matrix<UJMPMatrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    @Override
    public List<Matrix<UJMPMatrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    @NotNull
    private List<Matrix<UJMPMatrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<UJMPMatrix>> deltaParams = new ArrayList<>();
        InitialisationMethod m = InitialisationMethod.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new UJMPMatrix(m.initialisationValues(0, current, next), current, next));
        }
        return deltaParams;
    }
}
