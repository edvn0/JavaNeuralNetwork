package demos.implementations;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import demos.AbstractDemo;
import math.activations.ActivationFunction;
import math.activations.LeakyReluFunction;
import math.activations.TanhFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.OjAlgoFactory;
import neuralnetwork.inputs.NetworkInput;
import optimizers.ADAM;
import optimizers.StochasticGradientDescent;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxXOR extends AbstractDemo<OjAlgoMatrix> {
    private static final double[][] xorData = new double[][] { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
    private static final double[][] xorLabel = new double[][] { { 1 }, { 0 }, { 0 }, { 1 } };

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
        return Pair.of(50, 1);
    }

    @Override
    protected Triple<List<NetworkInput<OjAlgoMatrix>>, List<NetworkInput<OjAlgoMatrix>>, List<NetworkInput<OjAlgoMatrix>>> getData() {
        List<NetworkInput<OjAlgoMatrix>> data = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            double[] cData;
            double[] cLabel;
            int rd = ThreadLocalRandom.current().nextInt(xorData.length);
            cData = xorData[rd];
            cLabel = xorLabel[rd];
            data.add(new NetworkInput<OjAlgoMatrix>(new OjAlgoMatrix(cData, 2, 1), new OjAlgoMatrix(cLabel, 1, 1)));
        }
        Collections.shuffle(data);

        List<NetworkInput<OjAlgoMatrix>> train = data.subList(0, 7000);
        List<NetworkInput<OjAlgoMatrix>> validate = data.subList(7000, 9000);
        List<NetworkInput<OjAlgoMatrix>> test = data.subList(9000, 10000);

        return Triple.of(train, validate, test);
    }

    @Override
    protected NeuralNetwork<OjAlgoMatrix> createNetwork() {
        ActivationFunction<OjAlgoMatrix> f = new LeakyReluFunction<>(0.1);
        NeuralNetwork<OjAlgoMatrix> network = new NeuralNetwork<>(
                new NetworkBuilder<OjAlgoMatrix>(5).setFirstLayer(2).setLayer(35, f).setLayer(35, f)
                        .setLayer(20, new TanhFunction<>()).setLastLayer(1, f)
                        .setCostFunction(new MeanSquaredCostFunction<>())
                        .setEvaluationFunction(new ThresholdEvaluationFunction<>(0.1))
                        .setOptimizer(new ADAM<>(0.001, 0.9, 0.999)),
                new OjAlgoFactory(new int[] { 2, 35, 35, 20, 1 }, InitialisationMethod.XAVIER,
                        InitialisationMethod.SCALAR));
        return network;
    }

}
