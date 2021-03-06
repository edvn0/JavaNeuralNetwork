package neuralnetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.ParameterInitializer;

public class NetworkBuilder<M> {

    /**
     * A Builder for the Network.
     */
    protected int[] structure;
    protected int index;
    protected CostFunction<M> costFunction;
    protected EvaluationFunction<M> evaluationFunction;
    protected Optimizer<M> optimizer;
    protected int total;

    protected Map<Integer, ActivationFunction<M>> functionMap;
    protected List<Matrix<M>> weights;
    protected List<Matrix<M>> biases;
    protected ParameterInitializer<M> initialiser;

    public NetworkBuilder(int[] structure) {
        this.structure = structure;
        this.index = 0;
    }

    public NetworkBuilder(int s) {
        this.total = s;
        this.structure = new int[total];
        this.index = 0;
        this.functionMap = new HashMap<>();
    }

    public NeuralNetwork<M> compile() {
        if (costFunction == null || evaluationFunction == null || optimizer == null) {
            throw new IllegalArgumentException(
                    "You need to chose an implementation or implement a cost function, evaluation function and an optimizer.");
        }

        for (var e : functionMap.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) {
                throw new IllegalArgumentException("You need to provide correct implementations for the layers.");
            }
        }

        return new NeuralNetwork<M>(this);
    }

    public NetworkBuilder<M> setFirstLayer(final int i) {
        structure[index++] = i;
        this.functionMap.put(0, new DoNothingFunction<>());
        return this;
    }

    public NetworkBuilder<M> setOptimizer(Optimizer<M> o) {
        this.optimizer = o;
        return this;
    }

    public NetworkBuilder<M> setLayer(final int i, final ActivationFunction<M> f) {
        structure[index] = i;
        this.functionMap.put(index, f);
        index++;
        return this;
    }

    public NetworkBuilder<M> setInitialiser(final ParameterInitializer<M> pi) {
        this.initialiser = pi;
        return this;
    }

    public NetworkBuilder<M> setCostFunction(CostFunction<M> k) {
        this.costFunction = k;
        return this;
    }

    public NetworkBuilder<M> setEvaluationFunction(EvaluationFunction<M> f) {
        this.evaluationFunction = f;
        return this;
    }

    protected List<ActivationFunction<M>> getActivationFunctions() {

        var f = new ArrayList<ActivationFunction<M>>();

        if (this.functionMap.size() == this.structure.length) {
            // We do not care about this one, never gets evaluated.
            f.add(new DoNothingFunction<>());
            // We have one too many functions, one associated with the "first layer"
            // which in essence does not exist, we apply a linear function here.
            // However, this is never calculated.
            for (int i = 0; i < this.structure.length - 1; i++) {
                int index = i + 1;
                f.add(this.functionMap.get(index));
            }

        } else if (this.functionMap.size() + 1 == this.structure.length) {
            // We have supplied the builder with the correct amount of functions.
            for (int i = 0; i < this.structure.length; i++) {
                f.add(this.functionMap.get(i));
            }
        } else {
            throw new IllegalArgumentException("Not enough activation functions provided.");
        }
        return f;
    }

    public NetworkBuilder<M> setLastLayer(final int i, final ActivationFunction<M> f) {
        this.structure[index] = i;
        this.functionMap.put(total - 1, f);
        return this;
    }

    public void setWeights(List<Matrix<M>> weights) {
        this.weights = weights;
    }

    public void setBiases(List<Matrix<M>> biases) {
        this.biases = biases;
    }

}
