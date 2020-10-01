package neuralnetwork;

import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.error_functions.CostFunction;
import math.evaluation.EvaluationFunction;
import optimizers.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NetworkBuilder {

    /**
     * A Builder for the Network.
     */
    protected int[] structure;
    protected int index;
    protected CostFunction costFunction;
    protected EvaluationFunction evaluationFunction;
    protected Optimizer optimizer;
    protected int total;

    Map<Integer, ActivationFunction> functionMap;

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

    public NetworkBuilder compile() {
        if (costFunction == null || evaluationFunction == null || optimizer == null) {
            throw new IllegalArgumentException(
                    "You need to chose an implementation or implement a cost function, evaluation function and an optimizer.");
        }

        for (var e : functionMap.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) {
                throw new IllegalArgumentException("You need to provide correct implementations for the layers.");
            }
        }

        return this;
    }

    public NetworkBuilder setFirstLayer(final int i) {
        structure[index++] = i;
        this.functionMap.put(0, new DoNothingFunction());
        return this;
    }

    public NetworkBuilder setOptimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    public NetworkBuilder setLayer(final int i, final ActivationFunction f) {
        structure[index] = i;
        this.functionMap.put(index, f);
        index++;
        return this;
    }

    public NetworkBuilder setCostFunction(CostFunction k) {
        this.costFunction = k;
        return this;
    }

    public NetworkBuilder setEvaluationFunction(EvaluationFunction f) {
        this.evaluationFunction = f;
        return this;
    }

    protected List<ActivationFunction> getActivationFunctions() {

        var f = new ArrayList<ActivationFunction>();

        if (this.functionMap.size() == this.structure.length) {
            // We do not care about this one, never gets evaluated.
            f.add(new DoNothingFunction());
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

    public NetworkBuilder setLastLayer(final int i, final ActivationFunction f) {
        this.structure[index] = i;
        this.functionMap.put(total - 1, f);
        return this;
    }

}
