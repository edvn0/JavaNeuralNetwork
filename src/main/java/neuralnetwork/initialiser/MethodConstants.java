package neuralnetwork.initialiser;

import java.util.concurrent.ThreadLocalRandom;

public final class MethodConstants {

    private MethodConstants() {

    }

    public static final InitialisationMethod XAVIER = (previous, rows,
            cols) -> (2 * ThreadLocalRandom.current().nextDouble() - 1) * Math.sqrt(6d / (rows + cols));

    public static final InitialisationMethod RANDOM = (previous, rows, cols) -> ThreadLocalRandom.current()
            .nextDouble();

    public static final InitialisationMethod SCALAR = (a, b, c) -> 0.01;

    public static final InitialisationMethod ZERO = (a, b, c) -> 0;

}
