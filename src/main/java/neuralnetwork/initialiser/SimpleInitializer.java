package neuralnetwork.initialiser;

import java.util.ArrayList;
import java.util.List;
import math.linearalgebra.Matrix;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;

public class SimpleInitializer extends ParameterInitializer<SMatrix> {

    public SimpleInitializer(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(weightMethod, biasMethod);
    }

    @Override
    public void init(int[] sizes) {
        this.sizes = sizes.clone();
    }

    public List<Matrix<SMatrix>> getWeightParameters() {
        List<Matrix<SMatrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new SimpleMatrix(wM.initialisationValues(0, current, next)));
        }
        return weights;
    }

    public List<Matrix<SMatrix>> getBiasParameters() {
        List<Matrix<SMatrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new SimpleMatrix(bM.initialisationValues(0, current, 1)));
        }
        return biases;
    }

    @Override
    public List<Matrix<SMatrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    @Override
    public List<Matrix<SMatrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    protected List<Matrix<SMatrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<SMatrix>> deltaParams = new ArrayList<>();
        InitialisationMethod m = MethodConstants.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new SimpleMatrix(m.initialisationValues(0, current, next)));
        }
        return deltaParams;
    }

    @Override
    public Matrix<SMatrix> getFirstBias() {
        return new SimpleMatrix(this.bM.initialisationValues(0, this.sizes[0], 1));
    }

    @Override
    public String name() {
        return "SimpleIntializer";
    }

}
