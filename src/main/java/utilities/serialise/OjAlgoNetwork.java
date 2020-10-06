package utilities.serialise;

import java.util.Arrays;
import java.util.List;

import org.ojalgo.matrix.Primitive64Matrix;

import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitialiser;

public class OjAlgoNetwork {

    public static NeuralNetwork<Primitive64Matrix> create(List<Matrix<Primitive64Matrix>> weights,
            List<Matrix<Primitive64Matrix>> biases, int layers, int[] sizes,
            List<ActivationFunction<Primitive64Matrix>> functions, CostFunction<Primitive64Matrix> costFunc,
            Optimizer<Primitive64Matrix> optimiser, EvaluationFunction<Primitive64Matrix> evaluator) {

        NetworkBuilder<Primitive64Matrix> builder = new NetworkBuilder<>(layers);
        builder.setCostFunction(costFunc);
        builder.setEvaluationFunction(evaluator);
        builder.setOptimizer(optimiser);

        builder.setFirstLayer(sizes[0]);

        int[] paramSizes = new int[sizes.length - 1];
        for (int i = 1; i < sizes.length - 1; i++) {
            builder.setLayer(sizes[i], functions.get(i));
            paramSizes[i - 1] = sizes[i];
        }
        paramSizes[paramSizes.length - 1] = sizes[sizes.length - 1];

        builder.setLastLayer(sizes[sizes.length - 1], functions.get(functions.size() - 1));

        builder.setWeights(weights);
        builder.setBiases(biases);

        OjAlgoInitialiser initialiser = new OjAlgoInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR);
        initialiser.init(paramSizes);

        System.out.println(Arrays.toString(paramSizes));
        System.out.println(initialiser.getBiasParameters());

        NeuralNetwork<Primitive64Matrix> out = new NeuralNetwork<>(builder, initialiser);

        return out;
    }

}
