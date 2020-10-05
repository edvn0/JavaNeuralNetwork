package demos.implementations.ojalgo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.ojalgo.matrix.Primitive64Matrix;

import demos.AbstractDemo;
import math.activations.ActivationFunction;
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
import math.optimizers.StochasticGradientDescent;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxXOR extends AbstractDemo<Primitive64Matrix> {
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
                        .setEvaluationFunction(new ArgMaxEvaluationFunction<>())
                        .setOptimizer(new StochasticGradientDescent<>(0.1)),
                new OjAlgoInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR));
    }

}
