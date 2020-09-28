package math.linearalgebra.ujmp;

import lombok.extern.slf4j.Slf4j;
import math.linearalgebra.NeuralNetworkMatrix;
import org.ujmp.core.Matrix;

import java.util.function.Function;

@Slf4j
public class UJMPMatrix implements NeuralNetworkMatrix<UJMPMatrix, Double> {

    private final Matrix delegate;

    public UJMPMatrix(Matrix m) {
        this.delegate = m;
    }

    public UJMPMatrix(UJMPMatrix in) {
        this.delegate = Matrix.Factory.importFromArray(in.delegate.toDoubleArray());
    }

    public UJMPMatrix(double[] values, int rows, int cols) {
        if (values.length != rows * cols) {
            throw new IllegalArgumentException("The size of input array does not match rows and columns");
        }

        double[][] matrix = new double[rows][cols];
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                matrix[x][y] = values[y * rows + x];
            }
        }
        this.delegate = Matrix.Factory.importFromArray(matrix);
    }

    @Override
    public int rows() {
        return (int) this.delegate.getRowCount();
    }

    public int cols() {
        return (int) this.delegate.getColumnCount();
    }

    public Matrix getDelegate() {
        return delegate;
    }

    @Override
    public UJMPMatrix multiply(UJMPMatrix otherMatrix) {
        return new UJMPMatrix(this.delegate.mtimes(otherMatrix.delegate));
    }

    @Override
    public UJMPMatrix multiply(Double scalar) {
        return new UJMPMatrix(this.delegate.times(scalar));
    }

    @Override
    public UJMPMatrix add(UJMPMatrix in) {
        return new UJMPMatrix(this.delegate.plus(in.delegate));
    }

    @Override
    public UJMPMatrix add(Double in) {
        return new UJMPMatrix(this.delegate.plus(in));
    }

    @Override
    public UJMPMatrix subtract(Double in) {
        return new UJMPMatrix(this.delegate.minus(in));
    }

    @Override
    public UJMPMatrix divide(Double in) {
        return new UJMPMatrix(this.delegate.divide(in));
    }

    @Override
    public Double map(Function<UJMPMatrix, Double> mapping) {
        return mapping.apply(this);
    }


    @Override
    public UJMPMatrix mapElements(Function<double[][], UJMPMatrix> mapping) {
        return mapping.apply(this.delegate.toDoubleArray());
    }

    @Override
    public String toString() {
        return "UJMPMatrix{" +
                "delegate=\n" + delegate.toString() +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UJMPMatrix matrix = (UJMPMatrix) o;
        return delegate.equals(matrix.delegate);
    }

    @Override
    public int hashCode() {
        return delegate.hashCode();
    }
}
