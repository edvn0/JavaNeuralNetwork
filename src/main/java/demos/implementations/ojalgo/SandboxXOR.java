package demos.implementations.ojalgo;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.log4j.BasicConfigurator;
import org.ojalgo.matrix.Primitive64Matrix;

import demos.AbstractDemo;
import lombok.extern.slf4j.Slf4j;
import math.activations.ActivationFunction;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.costfunctions.CrossEntropyCostFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.optimizers.ADAM;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitialiser;
import neuralnetwork.inputs.NetworkInput;
import utilities.serialise.serialisers.OjAlgoSerializer;
import utilities.types.Pair;
import utilities.types.Triple;

@Slf4j
public class SandboxXOR extends AbstractDemo<Primitive64Matrix> {
    private static final double[][] xorData = new double[][] { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
    private static final double[][] xorLabel = new double[][] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

    @Override
    protected void demo() {
        BasicConfigurator.configure();
        final NeuralNetwork<Primitive64Matrix> network = createNetwork();
        final Triple<List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>> trainValidateTest = getData();
        final Pair<Integer, Integer> epochBatch = epochBatch();
        final TrainingMethod trainingMethod = networkTrainingMethod();
        final String outputPath = outputDirectory();

        network.display();

        long t1 = System.nanoTime();

        switch (trainingMethod) {
            case METRICS:
                network.trainWithMetrics(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right(), outputPath);
                break;
            case NORMAL:
                network.train(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right());
                break;
            case VERBOSE:
                network.trainVerbose(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right());
                break;
            default:
                break;

        }

        long t2 = System.nanoTime();

        double confusion = network.testEvaluation(trainValidateTest.getRight(), 50);
        double loss = network.testLoss(trainValidateTest.getRight());
        log.info("\nCorrectly evaluated {}% of the test set.\nFinal loss: {}", confusion * 100, loss);
        log.info("\nTotal time taken for training: {}.", (t2 - t1) * 1e-6);

        OjAlgoSerializer ser = new OjAlgoSerializer();
        ser.serialise(new File(
                "/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/resources/xor/serial.json"),
                network);
    }

    @Override
    protected String outputDirectory() {
        return "/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/resources/xor";
    }

    @Override
    protected TrainingMethod networkTrainingMethod() {
        return TrainingMethod.METRICS;
    }

    @Override
    protected Pair<Integer, Integer> epochBatch() {
        return Pair.of(50, 64);
    }

    @Override
    protected Triple<List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>> getData() {
        List<NetworkInput<Primitive64Matrix>> data = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            double[] cData;
            double[] cLabel;
            int rd = ThreadLocalRandom.current().nextInt(xorData.length);
            cData = xorData[rd];
            cLabel = xorLabel[rd];
            OjAlgoMatrix dataMatrix = new OjAlgoMatrix(cData, 2, 1);
            OjAlgoMatrix labelMatrix = new OjAlgoMatrix(cLabel, 2, 1);
            NetworkInput<Primitive64Matrix> in = new NetworkInput<>(dataMatrix, labelMatrix);
            data.add(in);
        }
        Collections.shuffle(data);

        List<NetworkInput<Primitive64Matrix>> train = data.subList(0, 7000);
        List<NetworkInput<Primitive64Matrix>> validate = data.subList(7000, 9000);
        List<NetworkInput<Primitive64Matrix>> test = data.subList(9000, 10000);

        return Triple.of(train, validate, test);
    }

    @Override
    protected NeuralNetwork<Primitive64Matrix> createNetwork() {
        ActivationFunction<Primitive64Matrix> f = new LeakyReluFunction<>(0.1);
        return new NeuralNetwork<>(
                new NetworkBuilder<Primitive64Matrix>(5).setFirstLayer(2).setLayer(3, f).setLayer(3, f).setLayer(2, f)
                        .setLastLayer(2, new SoftmaxFunction<>()).setCostFunction(new CrossEntropyCostFunction<>())
                        .setEvaluationFunction(new ThresholdEvaluationFunction<>(0.1))
                        .setOptimizer(new ADAM<>(0.01, 0.9, 0.999)),
                new OjAlgoInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR));
    }
}
