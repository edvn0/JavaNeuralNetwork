package neuralnetwork;

import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.error_functions.CostFunction;
import math.evaluation.EvaluationFunction;
import optimizers.Optimizer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    /**
     * A Builder for the Network.
     */
    protected int[] structure;
    protected int index;
    protected List<ActivationFunction> functions;
    protected CostFunction costFunction;
    protected EvaluationFunction evaluationFunction;
    protected Optimizer optimizer;

    public NetworkBuilder(int[] structure) {
        this.structure = structure;
        this.index = 0;
        this.functions = new ArrayList<>();
    }

    public NetworkBuilder(int s) {
        this.structure = new int[s];
        this.index = 0;
        this.functions = new ArrayList<>();
    }

    public NetworkBuilder setFirstLayer(final int i) {

        structure[index] = i;
        this.index++;

        functions.add(new DoNothingFunction());
        return this;
    }

    public NetworkBuilder setOptimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    public NetworkBuilder setLayer(final int i, final ActivationFunction f) {

        structure[index] = i;
        this.index++;
        functions.add(f);
        return this;
    }

    public NetworkBuilder setActivationFunction(ActivationFunction f) {
        this.functions.add(f);
        this.index++;
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

    protected ActivationFunction[] getActivationFunctions() {
        ActivationFunction[] f;
        f = new ActivationFunction[this.index];

        if (this.functions.size() == this.structure.length) {
            // We have one too many functions, one associated with the "first layer"
            // which in essence does not exist, we MNISTApply a linear function here.
            // However, this is never calculated.
            for (int i = 1; i < this.structure.length; i++) {
                f[i] = this.functions.get(i);
            }
            // We do not care about this one, never gets evaluated.
            f[0] = new DoNothingFunction();
        } else if (this.functions.size() + 1 == this.structure.length) {
            // We have supplied the builder with the correct amount of functions.
            for (int i = 0; i < this.structure.length; i++) {
                f[i] = this.functions.get(i);
            }
        } else {
            throw new IllegalArgumentException("Not enough activation functions provided.");
        }
        return f;
    }

    public NetworkBuilder setLastLayer(final int i, final ActivationFunction f) {
        this.structure[index] = i;
        this.index++;
        this.functions.add(f);
        return this;
    }

}
