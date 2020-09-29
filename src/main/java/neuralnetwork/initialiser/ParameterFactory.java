package neuralnetwork.initialiser;

import math.linearalgebra.Matrix;

import java.util.List;

public abstract class ParameterFactory<M> {

    protected int[] sizes;

    protected InitialisationMethod wM, bM;

    public ParameterFactory(int[] sizes, InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        this.sizes = sizes;
        this.wM = weightMethod;
        this.bM = biasMethod;
    }

    public abstract List<Matrix<M>> getWeightParameters();

    public abstract List<Matrix<M>> getBiasParameters();

    public abstract List<Matrix<M>> getDeltaWeightParameters();

    public abstract List<Matrix<M>> getDeltaBiasParameters();

}
