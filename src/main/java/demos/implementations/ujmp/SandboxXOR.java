package demos.implementations.ujmp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import demos.AbstractDemo;
import math.activations.ActivationFunction;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.ujmp.UJMPMatrix;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.UJMPInitialiser;
import neuralnetwork.inputs.NetworkInput;
import optimizers.StochasticGradientDescent;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxXOR extends AbstractDemo<org.ujmp.core.Matrix> {
    private static final double[][] xorData = new double[][] { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
    private static final double[][] xorLabel = new double[][] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

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
    protected Triple<List<NetworkInput<org.ujmp.core.Matrix>>, List<NetworkInput<org.ujmp.core.Matrix>>, List<NetworkInput<org.ujmp.core.Matrix>>> getData() {
        List<NetworkInput<org.ujmp.core.Matrix>> data = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            double[] cData;
            double[] cLabel;
            int rd = ThreadLocalRandom.current().nextInt(xorData.length);
            cData = xorData[rd];
            cLabel = xorLabel[rd];
            UJMPMatrix dataMatrix = new UJMPMatrix(cData, 2, 1);
            UJMPMatrix labelMatrix = new UJMPMatrix(cLabel, 2, 1);
            NetworkInput<org.ujmp.core.Matrix> in = new NetworkInput<>(dataMatrix, labelMatrix);
            data.add(in);
        }
        Collections.shuffle(data);

        List<NetworkInput<org.ujmp.core.Matrix>> train = data.subList(0, 7000);
        List<NetworkInput<org.ujmp.core.Matrix>> validate = data.subList(7000, 9000);
        List<NetworkInput<org.ujmp.core.Matrix>> test = data.subList(9000, 10000);

        return Triple.of(train, validate, test);
    }

    @Override
    protected NeuralNetwork<org.ujmp.core.Matrix> createNetwork() {
        ActivationFunction<org.ujmp.core.Matrix> f = new LeakyReluFunction<>(0.1);
        NeuralNetwork<org.ujmp.core.Matrix> network = new NeuralNetwork<org.ujmp.core.Matrix>(
                new NetworkBuilder<org.ujmp.core.Matrix>(5).setFirstLayer(2).setLayer(3, f).setLayer(3, f)
                        .setLayer(2, f).setLastLayer(2, new SoftmaxFunction<>())
                        .setCostFunction(new CrossEntropyCostFunction<>())
                        .setEvaluationFunction(new ArgMaxEvaluationFunction<>())
                        .setOptimizer(new StochasticGradientDescent<>(0.1)),
                new UJMPInitialiser(InitialisationMethod.XAVIER, InitialisationMethod.SCALAR));

        return network;
    }

}
