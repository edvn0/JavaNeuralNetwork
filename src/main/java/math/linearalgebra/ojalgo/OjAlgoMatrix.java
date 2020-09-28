package math.linearalgebra.ojalgo;

import math.linearalgebra.Matrix;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.MatrixUtilities;

import java.util.function.Function;

public class OjAlgoMatrix implements Matrix<OjAlgoMatrix> {

    private final Primitive64Matrix delegate;

    public OjAlgoMatrix(Primitive64Matrix in) {
        this.delegate = in;
    }

    public OjAlgoMatrix(double[] values, int rows, int cols) {
        double[][] nested = MatrixUtilities.fromFlat(values, rows, cols);
        this.delegate = Primitive64Matrix.FACTORY.rows(nested);
    }

    public OjAlgoMatrix(double[][] values, int rows, int cols) {
        double[] flat = MatrixUtilities.fromNested(values, rows, cols);
        this.delegate = Primitive64Matrix.FACTORY.rows(flat);
    }

    public OjAlgoMatrix(OjAlgoMatrix out) {
        this.delegate = out.delegate.copy().build();
    }

    public static OjAlgoMatrix identity(int rows, int cols) {
        return new OjAlgoMatrix(Primitive64Matrix.FACTORY.makeEye(rows, cols));
    }

    public static OjAlgoMatrix zeroes(int cols, int rows) {
        return new OjAlgoMatrix(Primitive64Matrix.FACTORY.make(cols, rows));
    }

    public static OjAlgoMatrix ones(int rows, int cols) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                data[i][j] = 1d;
            }
        }

        return new OjAlgoMatrix(data, rows, cols);
    }


    @Override
    public String toString() {
        return "Matrix<OjAlgoMatrix>{" +
                "delegate=" + delegate.toString() +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OjAlgoMatrix matrix = (OjAlgoMatrix) o;
        return delegate.equals(matrix.delegate);
    }

    @Override
    public int hashCode() {
        return delegate.hashCode();
    }

    @Override
    public int rows() {
        return 0;
    }

    @Override
    public int cols() {
        return 0;
    }

    @Override
    public OjAlgoMatrix multiply(OjAlgoMatrix otherMatrix) {
        return null;
    }

    @Override
    public OjAlgoMatrix multiply(double scalar) {
        return null;
    }

    @Override
    public OjAlgoMatrix add(OjAlgoMatrix in) {
        return null;
    }

    @Override
    public OjAlgoMatrix add(double in) {
        return null;
    }

    @Override
    public OjAlgoMatrix subtract(double in) {
        return null;
    }

    @Override
    public OjAlgoMatrix subtract(OjAlgoMatrix in) {
        return null;
    }

    @Override
    public OjAlgoMatrix divide(double in) {
        return null;
    }

    @Override
    public double map(Function<OjAlgoMatrix, Double> mapping) {
        return 0;
    }

    @Override
    public OjAlgoMatrix mapElements(Function<Double, Double> mapping) {
        return null;
    }

    @Override
    public double sum() {
        return 0;
    }

    @Override
    public double max() {
        return 0;
    }
}
