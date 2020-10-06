package neuralnetwork.initialiser;

import java.util.concurrent.ThreadLocalRandom;

public final class MethodConstants {

        private MethodConstants() {

        }

        public static final InitialisationMethod XAVIER = new InitialisationMethod() {

                @Override
                public String getName() {
                        return "XAVIER";
                }

                @Override
                public double calculateInitialisation(double previous, int rows, int cols) {
                        return (2 * ThreadLocalRandom.current().nextDouble() - 1) * Math.sqrt(6d / (rows + cols));
                }

        };

        public static final InitialisationMethod RANDOM = new InitialisationMethod() {

                @Override
                public String getName() {
                        return "RANDOM";
                }

                @Override
                public double calculateInitialisation(double previous, int rows, int cols) {
                        return ThreadLocalRandom.current().nextDouble();
                }

        };

        public static final InitialisationMethod SCALAR = new InitialisationMethod() {

                @Override
                public String getName() {
                        return "SCALAR";
                }

                @Override
                public double calculateInitialisation(double previous, int rows, int cols) {
                        return 0.01;
                }

        };

        public static final InitialisationMethod ZERO = new InitialisationMethod() {

                @Override
                public String getName() {
                        return "ZERO";
                }

                @Override
                public double calculateInitialisation(double previous, int rows, int cols) {
                        return 0;
                }

        };
}
