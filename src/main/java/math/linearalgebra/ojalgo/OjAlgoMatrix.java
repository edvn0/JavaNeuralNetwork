package math.linearalgebra.ojalgo;

import math.linearalgebra.NeuralNetworkMatrix;
import org.ojalgo.matrix.Primitive64Matrix;

import java.util.function.Function;

public class OjAlgoMatrix implements NeuralNetworkMatrix<OjAlgoMatrix, Double> {

    private final Primitive64Matrix delegate;

    public OjAlgoMatrix(Primitive64Matrix in) {
        this.delegate = in;
    }

    @Override
    public int rows() {
        return (int) this.delegate.countRows();
    }

    @Override
    public int cols() {
        return (int) this.delegate.countColumns();
    }

    @Override
    public OjAlgoMatrix multiply(OjAlgoMatrix otherMatrix) {
        return new OjAlgoMatrix(this.delegate.multiply(otherMatrix.delegate));
    }

    @Override
    public OjAlgoMatrix multiply(Double scalar) {
        return new OjAlgoMatrix(this.delegate.multiply(scalar));
    }

    @Override
    public OjAlgoMatrix add(OjAlgoMatrix in) {
        return new OjAlgoMatrix(this.delegate.add(in.delegate));
    }

    @Override
    public OjAlgoMatrix add(Double in) {
        return new OjAlgoMatrix(this.delegate.add(in));
    }

    @Override
    public OjAlgoMatrix subtract(Double in) {
        return new OjAlgoMatrix(this.delegate.subtract(in));
    }

    @Override
    public OjAlgoMatrix divide(Double in) {
        return new OjAlgoMatrix(this.delegate.divide(in));
    }

    @Override
    public Double map(Function<OjAlgoMatrix, Double> mapping) {
        return mapping.apply(this);
    }

    @Override
    public OjAlgoMatrix mapElements(Function<double[][], OjAlgoMatrix> mapping) {
        return mapping.apply(this.delegate.toRawCopy2D());
    }
}
