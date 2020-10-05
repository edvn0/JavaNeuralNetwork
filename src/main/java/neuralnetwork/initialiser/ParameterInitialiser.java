package neuralnetwork.initialiser;

import math.linearalgebra.Matrix;

import java.util.List;

public abstract class ParameterInitialiser<M> {

    protected int[] sizes;
    protected InitialisationMethod wM, bM;

    public ParameterInitialiser(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        this.wM = weightMethod;
        this.bM = biasMethod;
    }

    public void init(int[] sizes) {
        this.sizes = sizes.clone();
    }

    public abstract List<Matrix<M>> getWeightParameters();

    public abstract List<Matrix<M>> getBiasParameters();

    public List<Matrix<M>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    public List<Matrix<M>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    protected abstract List<Matrix<M>> getDeltaParameters(boolean isBias);

}
