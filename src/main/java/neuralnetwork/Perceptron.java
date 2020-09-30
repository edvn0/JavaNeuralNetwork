package neuralnetwork;

import java.util.Collection;

import math.linearalgebra.Matrix;

public interface Perceptron<M> {

    Matrix<M> predict(Matrix<M> in);

    void train(Collection<Matrix<M>> in);

    public String toString();

}
