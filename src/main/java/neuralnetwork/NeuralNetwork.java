package neuralnetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jetbrains.annotations.NotNull;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import me.tongfei.progressbar.ProgressBar;
import neuralnetwork.initialiser.ParameterInitialiser;
import neuralnetwork.inputs.NetworkInput;
import math.optimizers.Optimizer;
import utilities.NetworkUtilities;

/**
 * A class which can be both a single layer perceptron, and at the same time: an
 * artifical deep fully connected neural network. This implementation uses
 * matrices to solve the problem of learning and predicting on data.
 */
@Slf4j
public class NeuralNetwork<M> {

    // All activation functions for all layers
    private final List<ActivationFunction<M>> functions;
    // The error function to minimize.
    private final CostFunction<M> costFunction;
    // The function to evaluate the data set.
    private final EvaluationFunction<M> evaluationFunction;
    // The optimizer to be used
    private final Optimizer<M> optimizer;
    // Helper field to hold the total amount of layers
    private final int totalLayers;
    // The structure of the network
    private final int[] sizes;
    private final ParameterInitialiser<M> initialiser;
    // Weights and biases of the network
    private List<Matrix<M>> weights;
    private List<Matrix<M>> biases;
    private List<Matrix<M>> dW;
    private List<Matrix<M>> dB;
    private List<Matrix<M>> deltaWeights;
    private List<Matrix<M>> deltaBias;

    public NeuralNetwork(final NetworkBuilder<M> b, final ParameterInitialiser<M> parameterSupplier) {

        this.sizes = b.structure;
        this.functions = b.getActivationFunctions();
        this.costFunction = b.costFunction;
        this.evaluationFunction = b.evaluationFunction;
        this.totalLayers = b.total - 1;

        // Initialize the optimizer and the parameters.
        this.initialiser = parameterSupplier;
        this.initialiser.init(this.sizes);
        this.optimizer = b.optimizer;
        this.optimizer.initializeOptimizer(totalLayers, null, null);

        this.weights = parameterSupplier.getWeightParameters();
        this.dW = parameterSupplier.getDeltaWeightParameters();
        this.deltaWeights = parameterSupplier.getDeltaWeightParameters();

        this.biases = parameterSupplier.getBiasParameters();
        this.dB = parameterSupplier.getDeltaBiasParameters();
        this.deltaBias = parameterSupplier.getDeltaBiasParameters();
    }

    public NeuralNetwork(NeuralNetwork<M> n) {
        this.sizes = n.sizes;
        this.functions = n.functions;
        this.costFunction = n.costFunction;
        this.evaluationFunction = n.evaluationFunction;
        this.totalLayers = n.totalLayers;
        this.initialiser = n.initialiser;
        this.optimizer = n.optimizer;
        this.optimizer.initializeOptimizer(totalLayers, null, null);
        this.weights = initialiser.getWeightParameters();
        this.dW = initialiser.getDeltaWeightParameters();
        this.deltaWeights = initialiser.getDeltaWeightParameters();

        this.biases = initialiser.getBiasParameters();
        this.dB = initialiser.getDeltaBiasParameters();
        this.deltaBias = initialiser.getDeltaBiasParameters();
    }

    /**
     * Train the network with one example.
     *
     * @param input a {@link NetworkInput<M>} object to be trained on.
     */
    public void train(final NetworkInput<M> input) {
        evaluateTrainingExample(Collections.singletonList(input));
        learnFromDeltas();
    }

    /**
     * Back-propagates a data set and normalizes the deltas against the size of the
     * batch to be used in an optimizer.
     */
    protected void evaluateTrainingExample(final List<NetworkInput<M>> trainingExamples) {
        final int size = trainingExamples.size();
        final double inverse = 1d / size;

        for (final var data : trainingExamples) {
            final BackPropContainer<M> deltas = backPropagate(data);
            final List<Matrix<M>> deltaW = deltas.getDeltaWeights();
            final List<Matrix<M>> deltaB = deltas.getDeltaBiases();

            for (int j = 0; j < this.totalLayers; j++) {
                Matrix<M> newDeltaWeight = this.dW.get(j).add(deltaW.get(j).multiply(inverse));
                Matrix<M> newDeltaBias = this.dB.get(j).add(deltaB.get(j).multiply(inverse));
                this.dW.set(j, newDeltaWeight);
                this.dB.set(j, newDeltaBias);
            }
        }
    }

    /**
     * Updates weights and biases and resets the batch adjusted deltas.
     */
    void learnFromDeltas() {
        this.weights = this.optimizer.changeWeights(this.weights, this.dW);
        this.biases = this.optimizer.changeBiases(this.biases, this.dB);

        this.dB = this.initialiser.getDeltaBiasParameters();
        this.dW = this.initialiser.getDeltaWeightParameters();

    }

    private BackPropContainer<M> backPropagate(final NetworkInput<M> in) {
        this.deltaWeights = this.initialiser.getDeltaBiasParameters();
        this.deltaBias = this.initialiser.getDeltaWeightParameters();

        final List<Matrix<M>> activations = this.feedForward(in.getData());

        final Matrix<M> a = activations.get(activations.size() - 1);
        Matrix<M> deltaError = costFunction.applyCostFunctionGradient(a, in.getLabel());

        // Iterate over all layers, they are indexed by the last layer
        for (int k = totalLayers - 1; k >= 0; k--) {
            final Matrix<M> aCurr = activations.get(k + 1); // this layer
            final Matrix<M> aNext = activations.get(k); // Previous layer

            final Matrix<M> differentiate = this.functions.get(k + 1).derivativeOnInput(aCurr, deltaError);

            final Matrix<M> dB = differentiate;
            final Matrix<M> dW = differentiate.multiply(aNext.transpose());

            this.deltaBias.set(k, dB);
            this.deltaWeights.set(k, dW);

            deltaError = this.weights.get(k).transpose().multiply(differentiate);
        }

        return new BackPropContainer<>(this.deltaWeights, this.deltaBias);
    }

    /**
     * Feed forward inside the back propagation, mutates the actives list.
     *
     * @param starter Input NeuralNetworkMatrix<M><Matrix<M>>
     * @return
     */
    private List<Matrix<M>> feedForward(final Matrix<M> starter) {
        List<Matrix<M>> out = new ArrayList<>();
        Matrix<M> toPredict = starter;

        out.add(toPredict);
        for (int i = 0; i < this.totalLayers; i++) {
            final Matrix<M> x = this.weights.get(i).multiply(toPredict).add(this.biases.get(i));

            toPredict = this.functions.get(i + 1).function(x);
            out.add(toPredict);
        }
        return out;
    }

    /**
     * Feed the input through the network for classification.
     *
     * @param in VECTOR to predict
     * @return classified values.
     */
    public Matrix<M> predict(final Matrix<M> in) {
        Matrix<M> input = in; // row vector, from Nx1 to 1XN

        for (int i = 0; i < this.totalLayers; i++) {
            final Matrix<M> wI = this.weights.get(i).multiply(input);
            final Matrix<M> a = wI.add(this.biases.get(i));
            input = functions.get(i + 1).function(a);
        }

        return input;
    }

    private List<NetworkInput<M>> feedForwardData(final List<NetworkInput<M>> test) {
        final List<NetworkInput<M>> copy = new ArrayList<>();

        for (final NetworkInput<M> networkInput : test) {

            final Matrix<M> out = this.predict(networkInput.getData());
            final NetworkInput<M> newOut = new NetworkInput<M>(out, networkInput.getLabel());
            copy.add(newOut);
        }

        return copy;
    }

    public double testEvaluation(final List<NetworkInput<M>> imagesTest, final int size) {
        double avg = 0;
        final List<NetworkInput<M>> d = this.feedForwardData(imagesTest);
        for (int i = 0; i < size; i++) {
            avg += evaluate(d);
        }
        return avg / size;
    }

    public double testLoss(List<NetworkInput<M>> right) {
        return loss(feedForwardData(right));
    }

    double loss(List<NetworkInput<M>> data) {
        return this.costFunction.calculateCostFunction(data);
    }

    private double evaluate(final List<NetworkInput<M>> data) {
        return this.evaluationFunction.evaluatePrediction(data);
    }

    /**
     * Trains this network on training data, and validates on validation data. Uses
     * a {@link Optimizer} to optimize the gradient descent.
     *
     * @param training   a Collections object with {@link NetworkInput<M>} objects,
     *                   NetworkInput<M>.getData() is the data,
     *                   NetworkInput<M>.getLabel() is the label.
     * @param validation a Collections object with {@link NetworkInput<M>} objects,
     *                   NetworkInput<M>.getData() is the data,
     *                   NetworkInput<M>.getLabel() is the label.
     * @param epochs     how many iterations are we doing the descent for
     * @param batchSize  how big is the batch size, typically 32. See
     *                   https://stats.stackexchange.com/q/326663
     */
    public void train(@NotNull final List<NetworkInput<M>> training, @NotNull final List<NetworkInput<M>> validation,
            final int epochs, final int batchSize) {
        final List<List<NetworkInput<M>>> split = NetworkUtilities.batchSplitData(training, batchSize);
        for (int i = 0; i < epochs; i++) {
            // Randomize training sample.
            // randomisedBatchTraining(split);

            for (final var l : split) {
                this.evaluateTrainingExample(l);
                this.learnFromDeltas();
            }
        }
    }

    /**
     * Trains this network on training data, and validates on validation data. Uses
     * a {@link Optimizer} to optimize the gradient descent.
     * <p>
     * Displays a progress bar!
     *
     * @param training  a Collections object with {@link NetworkInput<M>} objects,
     *                  NetworkInput<M>.getData() is the data,
     *                  NetworkInput<M>.getLabel() is the label.
     * @param epochs    how many iterations are we doing the descent for
     * @param batchSize how big is the batch size, typically 32. See
     *                  https://stats.stackexchange.com/q/326663
     */
    public void trainVerbose(@NotNull final List<NetworkInput<M>> training, final List<NetworkInput<M>> validation,
            final int epochs, final int batchSize) {
        log.info("Started stochastic gradient descent, verbose mode on.%n");
        // How many times will we decrease the learning rate?
        final List<List<NetworkInput<M>>> split = NetworkUtilities.batchSplitData(training, batchSize);
        int info = epochs / 8;
        try (ProgressBar bar = new ProgressBar("Backpropagation", epochs)) {
            for (int i = 0; i < epochs; i++) {

                split.forEach(e -> {
                    this.evaluateTrainingExample(e);
                    this.learnFromDeltas();
                });

                if ((i + 1) % info == 0) {
                    // Feed forward the test data
                    final List<NetworkInput<M>> feedForwardData = this.feedForwardData(validation);
                    // Evaluate prediction with the interface EvaluationFunction.
                    double correct = evaluate(feedForwardData);
                    // Calculate cost/loss with the interface CostFunction
                    double loss = this.loss(feedForwardData);

                    log.info("\nThe network correctly evaluted \t {}\nThe network has a loss of {}.\n", correct * 100,
                            loss);
                }

                bar.step();
            }
        }
    }

    /**
     * Trains this network on training data, and validates on validation data. Uses
     * a {@link Optimizer} to optimize the gradient descent.
     *
     * @param training   a Collections object with {@link NetworkInput<M>} objects,
     *                   NetworkInput<M>.getData() is the data,
     *                   NetworkInput<M>.getLabel() is the label.
     * @param validation a Collections object with {@link NetworkInput<M>} objects,
     *                   NetworkInput<M>.getData() is the data,
     *                   NetworkInput<M>.getLabel() is the label.
     * @param batchSize  how big is the batch size, typically 32. See
     *                   https://stats.stackexchange.com/q/326663
     * @param path       To what path should the plots be printed?
     */
    public void trainWithMetrics(@NotNull final List<NetworkInput<M>> training,
            @NotNull final List<NetworkInput<M>> validation, final int epochs, final int batchSize, final String path) {

        long t1, t2;
        // Members which supply functionality to the plots.
        final NetworkMetrics metrics = new NetworkMetrics(training.get(0).getData().name());
        final List<List<NetworkInput<M>>> split = NetworkUtilities.batchSplitData(training, batchSize);

        // Feed forward the validation data prior to the batch descent
        // to establish a ground truth value
        final var ffD = this.feedForwardData(validation);
        // Evaluate prediction with the interface EvaluationFunction.
        double correct = evaluate(ffD);
        double loss = this.loss(ffD);
        metrics.initialPlotData(correct, loss);

        for (int i = 1; i <= epochs; i++) {

            // Calculates a batch of training data and update the deltas.
            t1 = System.nanoTime();
            split.stream().forEach(e -> {
                this.evaluateTrainingExample(e);
                this.learnFromDeltas();
            });
            t2 = System.nanoTime();

            // Feed forward the validation data
            Collections.shuffle(validation);
            final List<NetworkInput<M>> feedForwardData = this.feedForwardData(validation);

            // Evaluate prediction with the interface EvaluationFunction.
            correct = this.evaluate(feedForwardData);
            // Calculate cost/loss with the interface CostFunction
            loss = this.loss(feedForwardData);

            // Add the plotting data, x, y_1, y_2 to the
            // lists of xValues, correctValues, lossValues.
            metrics.addPlotData(i, correct, loss, (t2 - t1));

            if ((i - 1) % (epochs / 8) == 0) {
                log.info("\n {} / {} epochs are finished.\n Loss: \t {}", (i), epochs, loss);
            }
        }

        log.info("Outputting charts into " + path);
        try {
            metrics.present(path);
        } catch (final IOException e) {
            e.printStackTrace();
        }
        log.info("Charts outputted.");
    }

    private int[] weightDimensions(final int i) {
        return new int[] { (this.weights.get(i).rows()), (this.weights.get(i).cols()) };
    }

    public void display() {
        final StringBuilder b = new StringBuilder();
        b.append("\n").append("======================================================================").append("\n")
                .append("Network information and structure.").append("\n").append(String
                        .format("Input nodes: [%d]; Output nodes: [%d]%n%n", this.sizes[0], sizes[sizes.length - 1]));

        for (int i = 0; i < weights.size(); i++) {
            final int[] dims = weightDimensions(i);
            b.append(String.format("\t\tLayer %d : [%d X %d]%n", (i + 1), dims[0], dims[1])).append(
                    String.format("\t\tActivation function from this layer: %s", functions.get(i + 1).getName()))
                    .append("\n");
        }

        b.append("\n").append("The error function: ").append(this.costFunction).append("\n")
                .append("The evaluation function: ").append(this.evaluationFunction).append("\n")
                .append("The optimizer: ").append(this.optimizer).append("\n")
                .append("======================================================================");

        log.info(b.toString());
    }

    @Data
    @AllArgsConstructor
    static class BackPropContainer<U> {
        private List<Matrix<U>> deltaWeights;
        private List<Matrix<U>> deltaBiases;
    }

    protected List<Matrix<M>> getdB() {
        return this.dB;
    }

    protected List<Matrix<M>> getdW() {
        return this.dW;
    }

    protected Matrix<M> getSingleDb(int i) {
        return this.dB.get(i);
    }

    protected Matrix<M> getSingleDw(int i) {
        return this.dW.get(i);
    }
}
