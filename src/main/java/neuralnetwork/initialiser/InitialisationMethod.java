package neuralnetwork.initialiser;

import java.util.concurrent.ThreadLocalRandom;

public interface InitialisationMethod {
    InitialisationMethod XAVIER = (previous, rows, cols) -> (2 * ThreadLocalRandom.current().nextDouble() - 1) * Math.sqrt(6d / (rows + cols));

    InitialisationMethod RANDOM = (previous, rows, cols) -> ThreadLocalRandom.current().nextDouble();

    InitialisationMethod SCALAR = (a, b, c) -> 0.01;

    InitialisationMethod ZERO = (a, b, c) -> 0;

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
