package demos.implementations.ojalgo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.ojalgo.matrix.Primitive64Matrix;

import demos.AbstractDemo;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.costfunctions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitialiser;
import neuralnetwork.inputs.NetworkInput;
import math.optimizers.ADAM;
import math.optimizers.StochasticGradientDescent;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxMnist extends AbstractDemo<Primitive64Matrix> {

    @Override
    protected String outputDirectory() {
        return "/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/resources/mnist";
    }

    @Override
    protected TrainingMethod networkTrainingMethod() {
        return TrainingMethod.METRICS;
    }

    @Override
    protected Pair<Integer, Integer> epochBatch() {
        return Pair.of(9, 128);
    }

    @Override
    protected Triple<List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>> getData() {
        String test = "/mnist_test.csv";
        String train = "/mnist_train.csv";
        String path = "/Volumes/Toshiba 1,5TB/mnist";

        try (var trainInData = Files.lines(Paths.get(path + train));
                var testInData = Files.lines(Paths.get(path + test))) {
            List<NetworkInput<Primitive64Matrix>> trainData = trainInData.map(this::toMnist)
                    .collect(Collectors.toList());
            List<NetworkInput<Primitive64Matrix>> testData = testInData.map(this::toMnist).collect(Collectors.toList());
            int totalSize = trainData.size();
            int splitIndex = (int) (totalSize * 0.75);

            Collections.shuffle(trainData);
            List<NetworkInput<Primitive64Matrix>> trainingData = trainData.subList(0, splitIndex);
            List<NetworkInput<Primitive64Matrix>> validateData = trainData.subList(splitIndex, trainData.size());

            return Triple.of(trainingData, validateData, testData);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    protected NeuralNetwork<Primitive64Matrix> createNetwork() {
        var f = new LeakyReluFunction<Primitive64Matrix>(0.01);
        return new NeuralNetwork<>(
                new NetworkBuilder<Primitive64Matrix>(4).setFirstLayer(784).setLayer(10, f).setLayer(10, f)
                        .setLastLayer(10, new SoftmaxFunction<>()).setCostFunction(new CrossEntropyCostFunction<>())
                        .setEvaluationFunction(new ArgMaxEvaluationFunction<>())
                        .setOptimizer(new ADAM<>(0.01, 0.9, 0.999)), // new ADAM<>(0.01, 0.9, 0.999)),
                new OjAlgoInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR));
    }

    private NetworkInput<Primitive64Matrix> toMnist(String toMnist) {
        int imageSize = 28 * 28;
        int labelSize = 10;
        String labelString = toMnist.substring(0, 2).split(",")[0];
        String[] rest = toMnist.substring(2, toMnist.length()).split(",");

        int label = Integer.parseInt(labelString);
        double[] labels = new double[labelSize];
        labels[label] = 1;

        double[] values = new double[rest.length];

        for (int i = 0; i < imageSize; i++) {
            values[i] = Double.parseDouble(rest[i]) / 255;
        }

        return new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(values, imageSize, 1),
                new OjAlgoMatrix(labels, labelSize, 1));
    }
}
