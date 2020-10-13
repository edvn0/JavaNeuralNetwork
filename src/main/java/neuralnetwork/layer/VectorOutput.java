package neuralnetwork.layer;

import math.linearalgebra.Matrix;

public class VectorOutput<M> extends ZVector<M> {

    VectorOutput(Matrix<M> in) {
        super(in);
    }

    VectorOutput(VectorInput<M> in) {
        super(in.getMatrix());
    }
}