package math.linearalgebra.ujmp;

import lombok.extern.slf4j.Slf4j;
import math.linearalgebra.Matrix;
import utilities.MatrixUtilities;

import java.util.function.Function;

@Slf4j
public class UJMPMatrix implements Matrix<UJMPMatrix> {

    private final org.ujmp.core.Matrix delegate;

    public UJMPMatrix(org.ujmp.core.Matrix m) {
        this.delegate = m;
    }

    public UJMPMatrix(UJMPMatrix in) {
        this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(in.delegate.toDoubleArray());
    }

    public UJMPMatrix(double[] values, int rows, int cols) {
        if (values.length != rows * cols) {
            throw new IllegalArgumentException("The size of input array does not match rows and columns");
        }

        double[][] matrix = MatrixUtilities.fromFlat(values, rows, cols);
        this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(matrix);
    }

    public static UJMPMatrix identity(int rows, int cols) {
        return new UJMPMatrix(org.ujmp.core.Matrix.Factory.eye(rows, cols));
    }

    @Override
    public int rows() {
        return (int) this.delegate.getRowCount();
    }

    public int cols() {
        return (int) this.delegate.getColumnCount();
    }

    @Override
    public UJMPMatrix multiply(UJMPMatrix otherMatrix) {
        return new UJMPMatrix(this.delegate.mtimes(otherMatrix.delegate));
    }

    @Override
    public UJMPMatrix multiply(double scalar) {
        return new UJMPMatrix(this.delegate.times(scalar));
    }

    @Override
    public UJMPMatrix add(UJMPMatrix in) {
        return new UJMPMatrix(this.delegate.plus(in.delegate));
    }

    @Override
    public UJMPMatrix add(double in) {
        return new UJMPMatrix(this.delegate.plus(in));
    }

    @Override
    public UJMPMatrix subtract(double in) {
        return new UJMPMatrix(this.delegate.minus(in));
    }

    @Override
    public UJMPMatrix subtract(UJMPMatrix in) {
        return new UJMPMatrix(this.delegate.minus(in.delegate));
    }

    @Override
    public UJMPMatrix divide(double in) {
        return new UJMPMatrix(this.delegate.divide(in));
    }

    @Override
    public double map(Function<UJMPMatrix, Double> mapping) {
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
