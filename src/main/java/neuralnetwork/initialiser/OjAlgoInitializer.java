package neuralnetwork.initialiser;

import java.util.ArrayList;
import java.util.List;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import org.ojalgo.matrix.Primitive64Matrix;

public class OjAlgoInitializer extends ParameterInitializer<Primitive64Matrix> {

    public OjAlgoInitializer(InitialisationMethod weightMethod, InitialisationMethod biasMethod) {
        super(weightMethod, biasMethod);
    }

    @Override
    public void init(int[] sizes) {
        this.sizes = sizes.clone();
    }

    public List<Matrix<Primitive64Matrix>> getWeightParameters() {
        List<Matrix<Primitive64Matrix>> weights = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = this.sizes[i];
            weights.add(new OjAlgoMatrix(wM.initialisationValues(0, current, next)));
        }
        return weights;
    }

    public List<Matrix<Primitive64Matrix>> getBiasParameters() {
        List<Matrix<Primitive64Matrix>> biases = new ArrayList<>();
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            biases.add(new OjAlgoMatrix(bM.initialisationValues(0, current, 1)));
        }
        return biases;
    }

    @Override
    public List<Matrix<Primitive64Matrix>> getDeltaWeightParameters() {
        return getDeltaParameters(false);
    }

    @Override
    public List<Matrix<Primitive64Matrix>> getDeltaBiasParameters() {
        return getDeltaParameters(true);
    }

    protected List<Matrix<Primitive64Matrix>> getDeltaParameters(boolean isBias) {
        List<Matrix<Primitive64Matrix>> deltaParams = new ArrayList<>();
        InitialisationMethod m = MethodConstants.ZERO;
        for (int i = 0; i < this.sizes.length - 1; i++) {
            int current = this.sizes[i + 1];
            int next = isBias ? 1 : this.sizes[i];
            deltaParams.add(new OjAlgoMatrix(m.initialisationValues(0, current, next)));
        }
        return deltaParams;
    }

    @Override
    public Matrix<Primitive64Matrix> getFirstBias() {
        return new OjAlgoMatrix(this.bM.initialisationValues(0, this.sizes[0], 1));
    }

    @Override
    public String name() {
        return "OjAlgoInitializer";
    }

}
