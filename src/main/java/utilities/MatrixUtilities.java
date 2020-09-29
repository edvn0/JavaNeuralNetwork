package utilities;

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
}
