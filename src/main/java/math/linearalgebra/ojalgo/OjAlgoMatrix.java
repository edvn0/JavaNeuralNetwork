package math.linearalgebra.ojalgo;

import math.linearalgebra.Matrix;
import org.ojalgo.function.aggregator.Aggregator;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.MathUtilities;
import utilities.MatrixUtilities;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.Function;

public class OjAlgoMatrix implements Matrix<Primitive64Matrix> {

    /**
     *
     */
    private static final long serialVersionUID = -5585460400424961830L;
    private static final String NAME = "OjAlgo";
    protected Primitive64Matrix delegate;

    public OjAlgoMatrix(Primitive64Matrix in) {
        this.delegate = in;
    }

    public OjAlgoMatrix(double[] values, int rows, int cols) {
        double[][] nested = MatrixUtilities.fromFlat(values, rows, cols);
        this.delegate = Primitive64Matrix.FACTORY.rows(nested);
    }

    public OjAlgoMatrix(double[][] values, int rows, int cols) {
        this.delegate = Primitive64Matrix.FACTORY.rows(values);
    }

    public OjAlgoMatrix(OjAlgoMatrix out) {
        this.delegate = out.delegate.copy().build();
    }

    public OjAlgoMatrix(double[] values, MatrixType type, int rows, int cols) {
        switch (type) {
            case VECTOR:
                this.delegate = Primitive64Matrix.FACTORY.rows(values);
                break;
            case SQUARE:
                if (!MathUtilities.isSquare(values.length)) {
                    throw new IllegalArgumentException("Need to provide values of size NXN");
                }
                int sqrt = (int) Math.sqrt(values.length);
                this.delegate = Primitive64Matrix.FACTORY.rows(MatrixUtilities.fromFlat(values, sqrt, sqrt));
                break;
            case ONES:
                double[][] ones = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < rows; j++) {
                        ones[i][j] = 1;
                    }
                }
                this.delegate = Primitive64Matrix.FACTORY.rows(ones);
                break;
            case ZEROES:
                double[][] zeroes = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < rows; j++) {
                        zeroes[i][j] = 0;
                    }
                }
                this.delegate = Primitive64Matrix.FACTORY.rows(zeroes);
                break;
            case IDENTITY:
                this.delegate = Primitive64Matrix.FACTORY.makeEye(rows, cols);
                break;
            default:
                throw new IllegalArgumentException("Need to supply a matrix type");
        }
    }

    public OjAlgoMatrix(Primitive64Matrix matrix, int rows, int cols) {
        this.delegate = matrix;
    }

    @Override
    public String toString() {
        return delegate.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        OjAlgoMatrix matrix = (OjAlgoMatrix) o;
        return delegate.equals(matrix.delegate);
    }

    @Override
    public int hashCode() {
        return delegate.hashCode();
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
    public OjAlgoMatrix mapElements(Function<Double, Double> mapping) {
        double[][] elements = this.delegate.toRawCopy2D();
        double[][] out = new double[elements.length][elements[0].length];
        for (int i = 0; i < elements.length; i++) {
            for (int j = 0; j < elements[0].length; j++) {
                out[i][j] = mapping.apply(elements[i][j]);
            }
        }
        OjAlgoMatrix m = new OjAlgoMatrix(out, rows(), cols());
        return m;
    }

    @Override
    public double sum() {
        return this.delegate.aggregateAll(Aggregator.SUM);
    }

    @Override
    public double max() {
        double[] array = this.delegate.toRawCopy1D();
        double best = Double.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > best) {
                best = array[i];
            }
        }
        return best;
    }

    @Override
    public int argMax() {
        double[] array = this.delegate.toRawCopy1D();
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
    public OjAlgoMatrix transpose() {
        return new OjAlgoMatrix(this.delegate.transpose());
    }

    @Override
    public Primitive64Matrix delegate() {
        return this.delegate;
    }

    @Override
    public OjAlgoMatrix divide(Matrix<Primitive64Matrix> right) {
        double[][] array = this.delegate.toRawCopy2D();
        double[][] other = right.delegate().toRawCopy2D();
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                array[i][j] /= other[i][j];
            }
        }
        return new OjAlgoMatrix(array, rows(), cols());
    }

    @Override
    public OjAlgoMatrix maxVector() {
        double max = this.max();

        double[] values = new double[rows()];
        Arrays.fill(values, max);

        return new OjAlgoMatrix(values, rows(), 1);
    }

    @Override
    public OjAlgoMatrix zeroes(int rows, int cols) {
        double[][] zeroes = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                zeroes[i][j] = 0;
            }
        }
        return new OjAlgoMatrix(Primitive64Matrix.FACTORY.rows(zeroes));
    }

    @Override
    public OjAlgoMatrix ones(int rows, int cols) {
        double[][] ones = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                ones[i][j] = 1;
            }
        }
        return new OjAlgoMatrix(Primitive64Matrix.FACTORY.rows(ones));
    }

    @Override
    public OjAlgoMatrix identity(int rows, int cols) {
        return new OjAlgoMatrix(Primitive64Matrix.FACTORY.makeEye(rows, cols));
    }

    @Override
    public double square() {
        if (cols() != 1)
            throw new IllegalArgumentException("Trying to take the norm of matrix... sus.");

        return this.mapElements(e -> e * e).map(e -> e.sum());
    }

    @Override
    public OjAlgoMatrix hadamard(Matrix<Primitive64Matrix> otherMatrix) {
        double[][] elements = this.delegate.toRawCopy2D();
        double[][] out = otherMatrix.delegate().toRawCopy2D();
        for (int i = 0; i < elements.length; i++) {
            for (int j = 0; j < elements[0].length; j++) {
                out[i][j] *= (elements[i][j]);
            }
        }
        OjAlgoMatrix m = new OjAlgoMatrix(out, rows(), cols());
        return m;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public OjAlgoMatrix multiply(Matrix<Primitive64Matrix> otherMatrix) {
        return new OjAlgoMatrix(this.delegate.multiply(otherMatrix.delegate()));
    }

    @Override
    public OjAlgoMatrix multiply(double scalar) {
        return new OjAlgoMatrix(this.delegate.multiply(scalar));
    }

    @Override
    public OjAlgoMatrix add(Matrix<Primitive64Matrix> in) {
        return new OjAlgoMatrix(this.delegate.add(in.delegate()));
    }

    @Override
    public OjAlgoMatrix add(double in) {
        return new OjAlgoMatrix(this.delegate.add(in));
    }

    @Override
    public OjAlgoMatrix subtract(double in) {
        return new OjAlgoMatrix(this.delegate.subtract(in));
    }

    @Override
    public OjAlgoMatrix subtract(Matrix<Primitive64Matrix> in) {
        return new OjAlgoMatrix(this.delegate.subtract(in.delegate()));
    }

    @Override
    public OjAlgoMatrix divide(double in) {
        return new OjAlgoMatrix(this.delegate.divide(in));
    }

    @Override
    public double map(Function<Matrix<Primitive64Matrix>, Double> mapping) {
        return mapping.apply(this);
    }
}
