package neuralnetwork.initialiser;

public interface InitialisationMethod {

    String getName();

    double calculateInitialisation(double previous, int rows, int cols);

    default double[][] initialisationValues(double previous, int rows, int cols) {
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = calculateInitialisation(previous, rows, cols);
            }
        }
        return out;
    }
}
