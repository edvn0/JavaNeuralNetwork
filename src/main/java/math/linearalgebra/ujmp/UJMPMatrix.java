package math.linearalgebra.ujmp;

import lombok.extern.slf4j.Slf4j;
import math.linearalgebra.Matrix;
import org.ujmp.core.calculation.Calculation;
import utilities.MathUtilities;
import utilities.MatrixUtilities;

import java.util.function.Function;

@Slf4j
public class UJMPMatrix implements Matrix<UJMPMatrix> {

    protected final org.ujmp.core.Matrix delegate;

    public UJMPMatrix(org.ujmp.core.Matrix m) {
        this.delegate = m;
    }

    public UJMPMatrix(UJMPMatrix in) {
        this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(in.delegate.toDoubleArray());
    }

    public UJMPMatrix(double[] values, MatrixType type, int rows, int cols) {
        switch (type) {
            case VECTOR:
                this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(values);
                break;
            case SQUARE:
                if (!MathUtilities.isSquare(values.length)) {
                    throw new IllegalArgumentException("Need to provide values of size NXN");
                }
                int sqrt = (int) Math.sqrt(values.length);
                this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(MatrixUtilities.fromFlat(values, sqrt, sqrt));
                break;
            case ONES:
                this.delegate = org.ujmp.core.Matrix.Factory.ones(rows, cols);
                break;
            case ZEROES:
                this.delegate = org.ujmp.core.Matrix.Factory.zeros(rows, cols);
                break;
            case IDENTITY:
                this.delegate = org.ujmp.core.Matrix.Factory.eye(rows, cols);
                break;
            default:
                throw new IllegalArgumentException("Need to supply a matrix type");
        }
    }

    public UJMPMatrix(double[] values, int rows, int cols) {
        if (values.length != rows * cols) {
            throw new IllegalArgumentException("The size of input array does not match rows and columns");
        }

        double[][] matrix = MatrixUtilities.fromFlat(values, rows, cols);
        this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(matrix);
    }

    public UJMPMatrix(double[][] out, int rows, int cols) {
        this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(out);
    }

    public UJMPMatrix(Matrix<UJMPMatrix> out) {
        this.delegate = out.delegate().delegate;
    }


    @Override
    public int rows() {
        return (int) this.delegate.getRowCount();
    }

    @Override
    public int cols() {
        return (int) this.delegate.getColumnCount();
    }

    @Override
    public Matrix<UJMPMatrix> multiply(Matrix<UJMPMatrix> otherMatrix) {
        return new UJMPMatrix(this.delegate.mtimes(otherMatrix.delegate().delegate));
    }

    @Override
    public Matrix<UJMPMatrix> multiply(double scalar) {
        return new UJMPMatrix(this.delegate.times(scalar));
    }

    @Override
    public Matrix<UJMPMatrix> add(Matrix<UJMPMatrix> in) {
        return new UJMPMatrix(this.delegate.plus(in.delegate().delegate));
    }

    @Override
    public Matrix<UJMPMatrix> add(double in) {
        return new UJMPMatrix(this.delegate.plus(in));
    }

    @Override
    public Matrix<UJMPMatrix> subtract(double in) {
        return new UJMPMatrix(this.delegate.minus(in));
    }

    @Override
    public Matrix<UJMPMatrix> subtract(Matrix<UJMPMatrix> in) {
        return new UJMPMatrix(this.delegate.minus(in.delegate().delegate));
    }

    @Override
    public Matrix<UJMPMatrix> divide(double in) {
        return new UJMPMatrix(this.delegate.divide(in));
    }

    @Override
    public double map(Function<Matrix<UJMPMatrix>, Double> mapping) {
        return mapping.apply(this);
    }

    @Override
    public UJMPMatrix mapElements(Function<Double, Double> mapping) {
        double[][] elements = this.delegate.toDoubleArray();
        double[][] out = new double[elements.length][elements[0].length];
        for (int i = 0; i < elements.length; i++) {
            for (int j = 0; j < elements[0].length; j++) {
                out[i][j] = mapping.apply(elements[i][j]);
            }
        }
        return new UJMPMatrix(out, rows(), cols());
    }

    @Override
    public double sum() {
        return this.delegate.getValueSum();
    }

    @Override
    public double max() {
        return this.delegate.max(Calculation.Ret.NEW, 0).doubleValue();
    }

    @Override
    public UJMPMatrix transpose() {
        return new UJMPMatrix(this.delegate.transpose());
    }

    @Override
    public UJMPMatrix delegate() {
        return this;
    }

    @Override
    public Matrix<UJMPMatrix> divide(Matrix<UJMPMatrix> right) {
        return new UJMPMatrix(this.delegate.divide(right.delegate().delegate));
    }

    @Override
    public Matrix<UJMPMatrix> maxVector() {
        double max = this.max();

        double[][] vector = new double[cols()][1];
        for (int i = 0; i < cols(); i++) {
            vector[i][0] = max;
        }

        return new UJMPMatrix(vector, cols(), 1);
    }

    @Override
    public Matrix<UJMPMatrix> zeroes(int rows, int cols) {
        return new UJMPMatrix(org.ujmp.core.Matrix.Factory.zeros(rows, cols));
    }

    @Override
    public Matrix<UJMPMatrix> ones(int rows, int cols) {
        return new UJMPMatrix(org.ujmp.core.Matrix.Factory.ones(rows, cols));
    }

    @Override
    public Matrix<UJMPMatrix> identity(int rows, int cols) {
        return new UJMPMatrix(org.ujmp.core.Matrix.Factory.eye(rows, cols));
    }

    @Override
    public int argMax() {
        double[] array = this.delegate.toDoubleArray()[0];
        int argMax = -1;
        double best = Double.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > best) {
                best = array[i];
                argMax = i;
            }
        }

        return argMax;

    }

    @Override
    public String toString() {
        return delegate.toString();
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
