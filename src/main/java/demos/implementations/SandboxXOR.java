package demos.implementations;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import demos.AbstractDemo;
import math.activations.ActivationFunction;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.error_functions.BinaryCrossEntropyCostFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.ParameterInitialiser;
import neuralnetwork.inputs.NetworkInput;
import optimizers.ADAM;
import optimizers.StochasticGradientDescent;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxXOR extends AbstractDemo {
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
    protected Triple<List<NetworkInput>, List<NetworkInput>, List<NetworkInput>> getData() {
        List<NetworkInput> data = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            double[] cData;
            double[] cLabel;
            int rd = ThreadLocalRandom.current().nextInt(xorData.length);
            cData = xorData[rd];
            cLabel = xorLabel[rd];
            data.add(new NetworkInput(new OjAlgoMatrix(cData, 2, 1), new OjAlgoMatrix(cLabel, 2, 1)));
        }
        Collections.shuffle(data);

        List<NetworkInput> train = data.subList(0, 7000);
        List<NetworkInput> validate = data.subList(7000, 9000);
        List<NetworkInput> test = data.subList(9000, 10000);

        return Triple.of(train, validate, test);
    }

    @Override
    protected NeuralNetwork createNetwork() {
        ActivationFunction f = new LeakyReluFunction(0.1);
        NeuralNetwork network = new NeuralNetwork(new NetworkBuilder(5).setFirstLayer(2).setLayer(3, f).setLayer(3, f)
                .setLayer(2, f).setLastLayer(2, new SoftmaxFunction()).setCostFunction(new CrossEntropyCostFunction())
                .setEvaluationFunction(new ArgMaxEvaluationFunction()).setOptimizer(new StochasticGradientDescent(0.1)),
                new ParameterInitialiser(new int[] { 2, 3, 3, 2, 2 }, InitialisationMethod.XAVIER,
                        InitialisationMethod.SCALAR));
        return network;
    }

}
