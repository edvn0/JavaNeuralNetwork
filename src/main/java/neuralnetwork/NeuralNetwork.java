package neuralnetwork;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.val;
import lombok.extern.slf4j.Slf4j;
import math.activations.ActivationFunction;
import math.error_functions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import me.tongfei.progressbar.ProgressBar;
import neuralnetwork.initialiser.ParameterFactory;
import neuralnetwork.inputs.NetworkInput;
import optimizers.Optimizer;
import org.jetbrains.annotations.NotNull;
import utilities.NetworkUtilities;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Stream;

/**
 * A class which can be both a single layer perceptron, and at the same time: an
 * artifical deep fully connected neural network. This implementation uses
 * matrices to solve the problem of learning and predicting on data.
 */
@Slf4j
public class NeuralNetwork<M> implements Serializable {

    private static final long serialVersionUID = 7008674899707436812L;
    private static final String ERROR_MSG = "Something bad happened during deserialization";

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
    private final ParameterFactory<M> initialiser;
    // Weights and biases of the network
    private volatile List<Matrix<M>> weights;
    private volatile  List<Matrix<M>> biases;
    private volatile  List<Matrix<M>> dW;
    private volatile  List<Matrix<M>> dB;

    public NeuralNetwork(final NetworkBuilder<M> b, final ParameterFactory<M> factory) {

        this.sizes = b.structure;
        this.functions = b.getActivationFunctions();
        this.costFunction = b.costFunction;
        this.evaluationFunction = b.evaluationFunction;
        this.totalLayers = b.total - 1;

        // Initialize the optimizer and the parameters.
        this.initialiser = factory;
        this.optimizer = b.optimizer;
        this.optimizer.initializeOptimizer(totalLayers, null, null);

        this.weights = factory.getWeightParameters();
        this.dW = factory.getDeltaWeightParameters();

        this.biases = factory.getBiasParameters();
        this.dB = factory.getDeltaBiasParameters();
    }

    /**
     * Reads a .ser file or a path to a .ser file (with the extension excluded) to a
     * NeuralNetwork object.
     * <p>
     * E.g. /Users/{other paths}/NeuralNetwork_{LONG}_.ser works as well as
     * /Users/{other paths}/NeuralNetwork_{LONG}_
     *
     * @param path the full path to the file. does not require the .ser extension.
     * @return a deserialised object.
     * @throws IOException if file could not be found.
     */
    public static <M> NeuralNetwork<M> readObject(String path) throws IOException {
        NeuralNetwork<M> network = null;
        File file;
        path = (path.endsWith(".ser") ? path : path + ".ser");

        try (FileInputStream fs = new FileInputStream(file = new File(path));
                ObjectInputStream os = new ObjectInputStream(fs)) {

            network = (NeuralNetwork<M>) os.readObject();

            log.info("Completed deserialization from file:{}\n", file.getPath());
        } catch (final ClassNotFoundException e) {
            log.error(e.getMessage());
        }
        if (null != network) {
            return network;
        } else {
            log.error(ERROR_MSG);
            throw new IOException(ERROR_MSG);
        }
    }

    /**
     * Reads a .ser file or a path to a .ser file (with the extension excluded) to a
     * NeuralNetwork object.
     * <p>
     * E.g. /Users/{other paths}/NeuralNetwork_{LONG}_.ser works as well as
     * /Users/{other paths}/NeuralNetwork_{LONG}_
     *
     * @param file the file to read-
     * @return a deserialised network.
     * @throws IOException if file is not readable.
     */
    public static <M> NeuralNetwork<M> readObject(final File file) throws IOException {
        NeuralNetwork<M> neuralNetwork = null;
        try (FileInputStream fs = new FileInputStream(file); ObjectInputStream stream = new ObjectInputStream(fs)) {
            neuralNetwork = (NeuralNetwork<M>) stream.readObject();

            log.info("Completed deserialization, see file: {} \n", file.getAbsolutePath());
        } catch (final ClassNotFoundException e) {
            e.printStackTrace();
        }
        if (null != neuralNetwork) {
            return neuralNetwork;
        } else {
            log.error(ERROR_MSG);
            throw new IOException(ERROR_MSG);
        }
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
    private void evaluateTrainingExample(final List<NetworkInput<M>> trainingExamples) {
        final int size = trainingExamples.size();
        final double inverse = 1d/ size;

        for (final var data : trainingExamples) {
            final BackPropContainer<M> deltas = backPropagate(data);
            final List<Matrix<M>> deltaB = deltas.getDeltaBiases();
            final List<Matrix<M>> deltaW = deltas.getDeltaWeights();

            for (int j = 0; j < this.totalLayers - 1; j++) {
                this.dW.set(j, this.dW.get(j).add(deltaW.get(j).multiply(inverse)));
                this.dB.set(j, this.dB.get(j).add(deltaB.get(j).multiply(inverse)));
            }
        }
    }

    /**
     * Updates weights and biases and resets the batch adjusted deltas.
     */
    private synchronized void learnFromDeltas() {
        this.weights = this.optimizer.changeWeights(this.weights, this.dW);
        this.biases = this.optimizer.changeBiases(this.biases, this.dB);

        this.dW = this.initialiser.getDeltaWeightParameters();
        this.dB = this.initialiser.getDeltaBiasParameters();

    }

    private BackPropContainer<M> backPropagate(final NetworkInput<M> in) {
        final List<Matrix<M>> deltaBiases = this.initialiser.getDeltaBiasParameters();
        final List<Matrix<M>> deltaWeights = this.initialiser.getDeltaWeightParameters();
        
        final List<Matrix<M>> activations = this.feedForward(in.getData());
        // End feedforward

        final Matrix<M> a = activations.get(activations.size() - 1);
        Matrix<M> deltaError = costFunction.applyCostFunctionGradient(a, in.getLabel());

        // Iterate over all layers, they are indexed by the last layer
        for (int k = deltaBiases.size() - 1; k >= 0; k--) {
            final Matrix<M> aCurr = activations.get(k + 1); // this layer
            final Matrix<M> aNext = activations.get(k); // Previous layer

            final Matrix<M> differentiate = this.functions.get(k + 1).derivativeOnInput(aCurr, deltaError);
            
            final Matrix<M> dB = differentiate;
            final Matrix<M> dW = differentiate.multiply(aNext.transpose());
            
            deltaBiases.set(k, dB);
            deltaWeights.set(k, dW);
            
            deltaError = this.weights.get(k).transpose().multiply(differentiate);
        }

        return new BackPropContainer<>(deltaWeights, deltaBiases);
    }

    /**
     * Feed forward inside the back propagation, mutates the actives list.
     *
     * @param starter Input NeuralNetworkMatrix<Matrix>
     * @return
     */
    private List<Matrix<M>> feedForward(final Matrix<M> starter) {
        List<Matrix<M>> out = new ArrayList<>();
        Matrix<M> toPredict = starter;

        out.add(toPredict);
        for (int i = 0; i < this.totalLayers - 1; i++) {
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

        for (int i = 0; i < this.totalLayers - 1; i++) {
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

    public double evaluateTestData(final List<NetworkInput<M>> imagesTest, final int size) {
        double avg = 0;
        final List<NetworkInput<M>> d = this.feedForwardData(imagesTest);
        for (int i = 0; i < size; i++) {
            avg += this.evaluationFunction.evaluatePrediction(d);
        }
        return avg / size;
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

        final int info = epochs / 10;
        final List<List<NetworkInput<M>>> split = NetworkUtilities.batchSplitData(training, batchSize);
        for (int i = 0; i < epochs; i++) {
            // Randomize training sample.
            // randomisedBatchTraining(split);

            for (final var l : split) {
                this.evaluateTrainingExample(l);
                this.learnFromDeltas();
            }

            if (i % info == 0) {
                Collections.shuffle(validation);
                final List<NetworkInput<M>> l = this.feedForwardData(validation);
                final double loss = this.costFunction.calculateCostFunction(l);
                final double correct = this.evaluationFunction.evaluatePrediction(l) * 100;
                log.info("\nEpoch {}: Loss value of {}\n\tCorrect: {}% examples were classified correctly.\n\n", i, loss,
                        correct);
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
    public void trainVerbose(@NotNull final List<NetworkInput<M>> training, final int epochs, final int batchSize) {
        log.info("Started stochastic gradient descent, verbose mode on.%n");
        // How many times will we decrease the learning rate?
        final List<List<NetworkInput<M>>> split = NetworkUtilities.batchSplitData(training, batchSize);
        try (ProgressBar bar = new ProgressBar("Backpropagation", epochs)) {
            for (int i = 0; i < epochs; i++) {
                randomisedBatchTraining(split);
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
        final NetworkMetrics metrics = new NetworkMetrics();

        final var split = NetworkUtilities.batchSplitData(training, batchSize);

        // Feed forward the validation data prior to the batch descent
        // to establish a ground truth value
        final var ffD = this.feedForwardData(validation);
        // Evaluate prediction with the interface EvaluationFunction.
        double correct = this.evaluationFunction.evaluatePrediction(ffD);
        double loss = this.costFunction.calculateCostFunction(ffD);
        metrics.initialPlotData(correct, loss);

        for (int i = 0; i < epochs; i++) {

            // Calculates a batch of training data and update the deltas.
            t1 = System.nanoTime();
            randomisedBatchTraining(split);
            t2 = System.nanoTime();

            // Feed forward the test data
            final List<NetworkInput<M>> feedForwardData = this.feedForwardData(validation);

            // Evaluate prediction with the interface EvaluationFunction.
            correct = this.evaluationFunction.evaluatePrediction(feedForwardData);
            // Calculate cost/loss with the interface CostFunction
            loss = this.costFunction.calculateCostFunction(feedForwardData);

            // Add the plotting data, x, y_1, y_2 to the
            // lists of xValues, correctValues, lossValues.
            metrics.addPlotData(i + 1, correct, loss, (t2 - t1));

            if ((i + 1) % (epochs / 8) == 0) {
                log.info("\n {} / {} epochs are finished.\n Loss: \t {}", (i + 1), epochs, loss);
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

    private void randomisedBatchTraining(final List<List<NetworkInput<M>>> split) {
        /*
         * Collections.shuffle(split); for (List<NetworkInput<M>> networkInputs : split)
         * { Collections.shuffle(networkInputs); }
         */

        split.parallelStream().forEach(e -> {
            evaluateTrainingExample(e);
            learnFromDeltas();
        });
    }

    private int[] weightDimensions(final int i) {
        return new int[] { (this.weights.get(i).rows()), (this.weights.get(i).cols()) };
    }

    /**
     * A helper method to construct a batch of a List, "indexed" by the batch size.
     * For example: 0 to 10, 10 to 20, 20 to 30, etc...
     *
     * @param k         index into the list.
     * @param training  the list.
     * @param batchSize the batch size.
     * @return a slice of the list starting at k*batchSize.
     */
    private List<NetworkInput<M>> getBatch(final int k, final List<NetworkInput<M>> training, final int batchSize) {
        final int fromIx = k * batchSize;
        final int toIx = Math.min(training.size(), (k + 1) * batchSize);
        return Collections.unmodifiableList(training.subList(fromIx, toIx));
    }

    private List<List<NetworkInput<M>>> getSplitBatch(final int k, final List<List<NetworkInput<M>>> list,
            final int splitSize) {
        final int fromIx = k * splitSize;
        final int toIx = Math.min(list.size(), (k + 1) * splitSize);
        return Collections.unmodifiableList(list.subList(fromIx, toIx));
    }

    /**
     * A helper method to construct a batch in a Stream format, "indexed" by the
     * batch size. For example: 0 to 10, 10 to 20, 20 to 30, etc...
     *
     * @param k         index into the list.
     * @param training  the list.
     * @param batchSize the batch size.
     * @return a slice of the list starting at k*batchSize.
     */
    private Stream<NetworkInput<M>> getBatchStream(final int k, final List<NetworkInput<M>> training,
            final int batchSize) {
        final int fromIx = k * batchSize;
        final int toIx = Math.min(training.size(), (k + 1) * batchSize);
        return Collections.unmodifiableList(training.subList(fromIx, toIx)).parallelStream();
    }

    public void display() {
        final StringBuilder b = new StringBuilder();
        b.append("\n").append("======================================================================").append("\n")
                .append("Network information and structure.").append("\n").append(String
                        .format("Input nodes: [%d]; Output nodes: [%d]%n%n", this.sizes[0], sizes[sizes.length - 1]));

        for (int i = 0; i < weights.size(); i++) {
            final int[] dims = weightDimensions(i);
            b.append(String.format("\t\tLayer %d : [%d X %d]%n", (i + 1), dims[0], dims[1]))
                    .append(String.format("\t\tActivation function from this layer: %s", functions.get(i).getName())).append("\n");
        }

        b.append("\n")
                .append("The error function: ")
                .append(this.costFunction)
                .append("\n")
                .append("The evaluation function: ")
                .append(this.evaluationFunction)
                .append("\n")
                .append("The optimizer: ")
                .append(this.optimizer)
                .append("\n")
                .append("======================================================================");

        log.info(b.toString());
    }

    /**
     * Serialises this network. Outputs a file (.ser) with the date.
     *
     * @param path the path to the serialised file.
     */
    public void writeObject(final String path) {
        final String out = path.endsWith("/") ? path.substring(0, path.length() - 1) : path;
        String formattedDate;
        final SimpleDateFormat sdf = new SimpleDateFormat("dd-MM-yyyy", Locale.ENGLISH);
        formattedDate = sdf.format(new Date());

        try (final ObjectOutputStream fs = new ObjectOutputStream(
                new FileOutputStream(new File(out + "/NeuralNetwork_" + formattedDate + "_.ser")))) {

            fs.writeObject(this);

            log.info("Completed serialisation.");
        } catch (final IOException e) {
            log.error(e.getMessage());
        }
    }

    @Data
    @AllArgsConstructor
    private static class BackPropContainer<M> {
        private List<Matrix<M>> deltaWeights;
        private List<Matrix<M>> deltaBiases;
    }

}
