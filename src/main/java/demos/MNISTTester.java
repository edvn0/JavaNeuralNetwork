package demos;

import lombok.extern.slf4j.Slf4j;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.MatrixSupplier;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.inputs.NetworkInput;
import optimizers.ADAM;
import utilities.NetworkUtilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
public class MNISTTester {

    public static void main(final String[] args) throws IOException {
        //if (args.length != 3)
        //    throw new IllegalArgumentException("Need to supply epochs, batches, output path.");

        final int epochs = 20;// Integer.parseInt(args[0]);
        final int batch = 64;//;Integer.parseInt(args[1]);
        final String output = "E:\\Programming\\Git\\JavaNeuralNetwork\\src\\main\\resources\\output";//args[2];

        NeuralNetwork network = NeuralNetwork.of(MatrixSupplier.UJMP,
                new NetworkBuilder(3)
                        .setFirstLayer(784)
                        .setLayer(35, new LeakyReluFunction(0.01))
                        .setLastLayer(10, new SoftmaxFunction())
                        .setCostFunction(new CrossEntropyCostFunction())
                        .setEvaluationFunction(new ArgMaxEvaluationFunction())
                        .setOptimizer(new ADAM(0.001, 0.9, 0.999))
        );

        List<NetworkInput> imagesTrain = generateDataFromCSV("C:\\Users\\edvin\\Downloads\\MNIST\\mnist_train.csv");
        List<NetworkInput> imagesValidate = generateDataFromCSV("C:\\Users\\edvin\\Downloads\\MNIST\\mnist_test.csv");

        Collections.shuffle(imagesTrain);
        Collections.shuffle(imagesValidate);

        final List<NetworkInput> imagesTest = imagesTrain.subList(0, (int) (imagesTrain.size() * 0.1));

        imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

        network.trainWithMetrics(imagesTrain, imagesValidate, epochs, batch, output);

        double correct = network.evaluateTestData(imagesTest, 100);
        log.info("Correct evaluation percentage: {}", correct);
        network.writeObject(output);
        System.exit(0);
    }

    private static List<NetworkInput> generateDataFromCSV(final String path) {
        try (var out = Files.lines(Paths.get(path))) {
            return out.map(e -> e.split(",")).map(NetworkUtilities::MNISTApply).collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return Collections.emptyList();
    }
}
