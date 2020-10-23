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

    static InitialisationMethod get(String method) {
        switch (method) {
            case "ZERO":
                return MethodConstants.ZERO;
            case "SCALAR":
                return MethodConstants.SCALAR;
            case "XAVIER":
                return MethodConstants.XAVIER;
            case "RANDOM":
                return MethodConstants.RANDOM;
            default:
                throw new IllegalArgumentException("Unsupported init method.");
        }

    }
}
