package utilities;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import java.util.function.UnaryOperator;

public class MatrixUtilities {

    public static double[] fromNested(double[][] input, int rows, int cols) {
        double[] flat = new double[rows * cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[j * rows + i] = input[i][j];
            }
        }
        return flat;
    }

    public static double[][] fromFlat(double[] input, int rows, int cols) {
        double[][] nested = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                nested[i][j] = input[j * rows + i];
            }
        }
        return nested;
    }

    public static double[] toArray(Matrix b) {
        return b.transpose().toDoubleArray()[0];
    }

    public static Matrix toMatrix(double[][] doubles) {
        return Matrix.Factory.importFromArray(doubles);
    }

    /**
     * Creates a vector, either row or column based on the dim. If dim is true, a
     * column vector will be create, else a row vector.
     *
     * @param doubles the data from which the {@link Matrix} will be created.
     * @param dim     column or row vector?
     * @return a new {@link Matrix} based in doubles.
     */
    public static Matrix toMatrix(double[] doubles, boolean dim) {
        if (dim) {
            double[][] from = new double[doubles.length][1];
            int i = 0;
            for (double d : doubles) {
                from[i][0] = d;
            }
            return Matrix.Factory.importFromArray(from);
        } else {
            return Matrix.Factory.importFromArray(doubles);
        }
    }

    public static Matrix map(Matrix in, UnaryOperator<Double> t) {
        double[][] values = in.toDoubleArray();
        double[][] out = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                double temp = values[i][j];
                double applied = t.apply(temp);
                out[i][j] = applied;
            }
        }
        return Matrix.Factory.importFromArray(out);
    }

    public static int argMax(Matrix input) {
        Matrix argMax = input.indexOfMax(Ret.NEW, 0);
        return argMax.intValue();
    }
}
