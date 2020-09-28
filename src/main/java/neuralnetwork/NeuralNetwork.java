package neuralnetwork;

import lombok.extern.slf4j.Slf4j;
import math.activations.ActivationFunction;
import math.activations.LeakyReluFunction;
import math.activations.TanhFunction;
import math.error_functions.CostFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.ThreshHoldEvaluationFunction;
import math.linearalgebra.MatrixSupplier;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import me.tongfei.progressbar.ProgressBar;
import neuralnetwork.initialiser.BiasInitialiser;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.Initialiser;
import neuralnetwork.initialiser.WeightInitialiser;
import neuralnetwork.inputs.NetworkInput;
import optimizers.ADAM;
import optimizers.Optimizer;
import org.apache.log4j.BasicConfigurator;
import org.jetbrains.annotations.NotNull;
import utilities.NetworkUtilities;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;

/**
 * A class which can be both a single layer perceptron, and at the same time: an
 * artifical deep fully connected neural network. This implementation uses
 * matrices to solve the problem of learning and predicting on data.
 */
@Slf4j
public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 7008674899707436812L;
    private static final String ERROR_MSG = "Something bad happened during deserialization";

    static {
        BasicConfigurator.configure();
    }

    // All activation functions for all layers
    private final ActivationFunction[] functions;
    // The error function to minimize.
    private final CostFunction costFunction;
    // The function to evaluate the data set.
    private final EvaluationFunction evaluationFunction;
    // The optimizer to be used
    private final Optimizer optimizer;
    // Helper field to hold the total amount of layers
    private final int totalLayers;
    // The structure of the network
    private final int[] sizes;

    private final Initialiser wInitialiser;
    private final Initialiser bInitialiser;

    // Weights and biases of the network
    private volatile OjAlgoMatrix[] weights;
    private volatile OjAlgoMatrix[] biases;
    private volatile OjAlgoMatrix[] dW;
    private volatile OjAlgoMatrix[] dB;

    public NeuralNetwork(MatrixSupplier s, NetworkBuilder b) {
        this.sizes = b.structure;
        this.functions = b.getActivationFunctions();
        this.costFunction = b.costFunction;
        this.evaluationFunction = b.evaluationFunction;
        this.totalLayers = sizes.length;

        // Initialize the optimizer and the parameters.
        this.optimizer = b.optimizer;
        this.optimizer.initializeOptimizer(this.totalLayers);

        this.wInitialiser = new WeightInitialiser(this.sizes, InitialisationMethod.XAVIER);
        this.bInitialiser = new BiasInitialiser(this.sizes, InitialisationMethod.SCALAR);

        var weights = wInitialiser.createMatrices();
        var biases = bInitialiser.createMatrices();

        this.weights =  weights.getParameters();
        this.dW = weights.getDeltaParameters();

        this.biases = biases.getParameters();
        this.dB = biases.getDeltaParameters();
    }

    /**
     * Defaults a neural network with the structure [[in * 1, linear], [35*1, leaky
     * Relu], [35*1, leaky Relu], [out * 1, Tanh]] with ADAM as optimizer, MSE as
     * error function, a threshhold accuracy of |y-y'| < 0.01 (alternatively cross
     * entropy for error function and arg max evaluator)
     *
     * @param in                         input data size
     * @param out                        output data size
     * @param regressionOrClassification classification or regression? I.e, threshold or cross
     *                                   entropy
     */
    public static NeuralNetwork standardLearner(MatrixSupplier s, int in, int out, boolean regressionOrClassification) {
        return new NeuralNetwork(s, new NetworkBuilder(in).setFirstLayer(out).setLayer(35, new LeakyReluFunction(0.01))
                .setLayer(35, new LeakyReluFunction(0.01)).setLastLayer(10, new TanhFunction())
                .setCostFunction(
                        regressionOrClassification ? new MeanSquaredCostFunction() : new CrossEntropyCostFunction())
                .setEvaluationFunction(regressionOrClassification ? new ThreshHoldEvaluationFunction(0.01)
                        : new ArgMaxEvaluationFunction())
                .setOptimizer(new ADAM(0.001, 0.9, 0.999)));
    }

    public static NeuralNetwork of(MatrixSupplier s, NetworkBuilder b) {
        return new NeuralNetwork(s,b);
    }

    private double xavierInitialization(final int prev) {
        return ThreadLocalRandom.current().nextGaussian() * (Math.sqrt(2) / Math.sqrt(prev));
    }

    /**
     * Train the network with one example.
     *
     * @param input a {@link NetworkInput} object to be trained on.
     */
    public void train(final NetworkInput input) {
        evaluateTrainingExample(Collections.singletonList(input));
        learnFromDeltas();
    }

    /**
     * Back-propagates a data set and normalizes the deltas against the size of the
     * batch to be used in an optimizer.
     */
    private void evaluateTrainingExample(final List<NetworkInput> trainingExamples) {
        final int size = trainingExamples.size();

        for (final var data : trainingExamples) {
            final List<OjAlgoMatrix[]> deltas = backPropagate(data);
            final OjAlgoMatrix[] deltaB = deltas.get(0);
            final OjAlgoMatrix[] deltaW = deltas.get(1);

            for (int j = 0; j < this.totalLayers - 1; j++) {
                this.dW[j] = this.dW[j].add(deltaW[j].multiply(1d / size));
                this.dB[j] = this.dB[j].add(deltaB[j].multiply(1d / size));
            }
        }
    }

    /**
     * Evaluates one example for multi threaded gradient descent.
     */
    private void evaluateTrainingExample(final NetworkInput ni) {
        final List<OjAlgoMatrix[]> deltas = backPropagate(ni);
        final OjAlgoMatrix[] deltaB = deltas.get(0);
        final OjAlgoMatrix[] deltaW = deltas.get(1);

        for (int j = 0; j < this.totalLayers - 1; j++) {
            dW[j] = dW[j].add(deltaW[j]);
            dB[j] = dB[j].add(deltaB[j]);
        }
    }

    /**
     * Updates weights and biases and resets the batch adjusted deltas.
     */
    private synchronized void learnFromDeltas() {
        this.weights = this.optimizer.changeWeights(this.weights, this.dW);
        this.biases = this.optimizer.changeBiases(this.biases, this.dB);

        this.dB = bInitialiser.clearDeltas();
        this.dW = wInitialiser.clearDeltas();

    }

    private synchronized List<OjAlgoMatrix[]> backPropagate(NetworkInput in) {

        final List<OjAlgoMatrix[]> totalDeltas = new ArrayList<>();
        final List<OjAlgoMatrix> activations = new ArrayList<>();

        OjAlgoMatrix[] deltaBiases = bInitialiser.deltaParameters(this.biases);
        OjAlgoMatrix[] deltaWeights = wInitialiser.deltaParameters(this.weights);

        // Alters all arrays and lists.
        this.feedForward(in.getData(), activations);
        // End feedforward

        OjAlgoMatrix a = activations.get(activations.size() - 1);
        OjAlgoMatrix deltaError = costFunction.applyCostFunctionGradient(a, in.getLabel());

        // Iterate over all layers, they are indexed by the last layer (here given b
        for (int k = deltaBiases.length - 1; k >= 0; k--) {
            final OjAlgoMatrix aCurr = activations.get(k + 1); // this layer
            final OjAlgoMatrix aNext = activations.get(k); // Previous layer
            OjAlgoMatrix differentiate = this.functions[k + 1].derivativeOnInput(aCurr, deltaError);

            deltaBiases[k] = differentiate;
            deltaWeights[k] = differentiate.multiply(aNext.transpose());

            deltaError = this.weights[k].transpose().multiply(differentiate);
        }

        totalDeltas.add(deltaBiases);
        totalDeltas.add(deltaWeights);

        return totalDeltas;
    }

    /**
     * Feed forward inside the back propagation, mutates the actives list.
     *
     * @param starter Input NeuralNetworkMatrix<Matrix>
     * @param actives activations list
     */
    private void feedForward(final OjAlgoMatrix starter, final List<OjAlgoMatrix> actives) {
        OjAlgoMatrix toPredict = starter;

        actives.add(toPredict);
        for (int i = 0; i < this.totalLayers - 1; i++) {
            final OjAlgoMatrix x = this.weights[i].multiply(toPredict).add(this.biases[i]);

            toPredict = this.functions[i + 1].function(x);
            actives.add(toPredict);
        }
    }

    /**
     * Predict a single example input data.
     *
     * @param in {@link Matrix <Matrix>} a NeuralNetworkMatrix<Matrix> to feed forward.
     * @return a classification of {@link Matrix <Matrix>}
     */
    public OjAlgoMatrix predict(final OjAlgoMatrix in) {
        return feedForward(in);
    }

    /**
     * Feed the input through the network for classification.
     *
     * @param in values to predict
     * @return classified values.
     */
    private OjAlgoMatrix feedForward(final OjAlgoMatrix in) {
        OjAlgoMatrix input = in;
        for (int i = 0; i < this.totalLayers - 1; i++) {
            input = functions[i + 1].function(this.weights[i].multiply(input).add(this.biases[i]));
        }
        return input;
    }

    private List<NetworkInput> feedForwardData(final List<NetworkInput> test) {
        final List<NetworkInput> copy = new ArrayList<>();

        for (final NetworkInput networkInput : test) {

            final OjAlgoMatrix out = this.feedForward(networkInput.getData());
            final NetworkInput newOut = new NetworkInput(out, networkInput.getLabel());
            copy.add(newOut);
        }

        return copy;
    }

    public double evaluateTestData(final List<NetworkInput> imagesTest, int size) {
        double avg = 0;
        List<NetworkInput> d = this.feedForwardData(imagesTest);
        for (int i = 0; i < size; i++) {
            avg += this.evaluationFunction.evaluatePrediction(d);
        }
        return avg / size;
    }

    /**
     * Trains this network on training data, and validates on validation data. Uses
     * a {@link Optimizer} to optimize the gradient descent.
     *
     * @param training   a Collections object with {@link NetworkInput} objects,
     *                   NetworkInput.getData() is the data, NetworkInput.getLabel()
     *                   is the label.
     * @param validation a Collections object with {@link NetworkInput} objects,
     *                   NetworkInput.getData() is the data, NetworkInput.getLabel()
     *                   is the label.
     * @param epochs     how many iterations are we doing the descent for
     * @param batchSize  how big is the batch size, typically 32. See
     *                   https://stats.stackexchange.com/q/326663
     */
    public void train(@NotNull final List<NetworkInput> training, @NotNull final List<NetworkInput> validation,
                      final int epochs, final int batchSize) {

        int info = epochs / 10;
        List<List<NetworkInput>> split = NetworkUtilities.batchSplitData(training, batchSize);
        for (int i = 0; i < epochs; i++) {
            // Randomize training sample.
            randomisedBatchTraining(split);

            if (i % info == 0) {
                List<NetworkInput> l = this.feedForwardData(validation);
                double loss = this.costFunction.calculateCostFunction(l);
                double correct = this.evaluationFunction.evaluatePrediction(l) * 100;
                log.info("Epoch {}: Loss value of {}\n\tCorrect: {}% examples were classified correctly.\n\n", i, loss,
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
     * @param training  a Collections object with {@link NetworkInput} objects,
     *                  NetworkInput.getData() is the data, NetworkInput.getLabel()
     *                  is the label.
     * @param epochs    how many iterations are we doing the descent for
     * @param batchSize how big is the batch size, typically 32. See
     *                  https://stats.stackexchange.com/q/326663
     */
    public void trainVerbose(@NotNull final List<NetworkInput> training, final int epochs, final int batchSize) {
        log.info("Started stochastic gradient descent, verbose mode on.%n");
        // How many times will we decrease the learning rate?
        List<List<NetworkInput>> split = NetworkUtilities.batchSplitData(training, batchSize);
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
     * @param training   a Collections object with {@link NetworkInput} objects,
     *                   NetworkInput.getData() is the data, NetworkInput.getLabel()
     *                   is the label.
     * @param validation a Collections object with {@link NetworkInput} objects,
     *                   NetworkInput.getData() is the data, NetworkInput.getLabel()
     *                   is the label.
     * @param batchSize  how big is the batch size, typically 32. See
     *                   https://stats.stackexchange.com/q/326663
     * @param path       To what path should the plots be printed?
     */
    public void trainWithMetrics(@NotNull final List<NetworkInput> training,
                                 @NotNull final List<NetworkInput> validation, final int epochs, final int batchSize,
                                 String path) {

        long t1, t2;
        // Members which supply functionality to the plots.
        NetworkMetrics metrics = new NetworkMetrics();

        var split = NetworkUtilities.batchSplitData(training, batchSize);

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
            final List<NetworkInput> feedForwardData = this.feedForwardData(validation);

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
        } catch (IOException e) {
            e.printStackTrace();
        }
        log.info("Charts outputted.");
    }

    private void randomisedBatchTraining(List<List<NetworkInput>> split) {
        /*Collections.shuffle(split);
        for (List<NetworkInput> networkInputs : split) {
            Collections.shuffle(networkInputs);
        }*/

        split.parallelStream().forEach(e -> {
            evaluateTrainingExample(e);
            learnFromDeltas();
        });
    }

    public void train() {

    }

    private int[] weightDimensions(int i) {
        return new int[]{(this.weights[i].rows()), (this.weights[i].cols())};
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
    private List<NetworkInput> getBatch(final int k, final List<NetworkInput> training, int batchSize) {
        int fromIx = k * batchSize;
        int toIx = Math.min(training.size(), (k + 1) * batchSize);
        return Collections.unmodifiableList(training.subList(fromIx, toIx));
    }

    private List<List<NetworkInput>> getSplitBatch(final int k, final List<List<NetworkInput>> list, int splitSize) {
        int fromIx = k * splitSize;
        int toIx = Math.min(list.size(), (k + 1) * splitSize);
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
    private Stream<NetworkInput> getBatchStream(final int k, final List<NetworkInput> training, int batchSize) {
        int fromIx = k * batchSize;
        int toIx = Math.min(training.size(), (k + 1) * batchSize);
        return Collections.unmodifiableList(training.subList(fromIx, toIx)).parallelStream();
    }

    public void display() {
        StringBuilder b = new StringBuilder();
        b.append("======================================================================").append("\n")
                .append("Network information and structure.").append("\n")
                .append(String.format("Input nodes: [%d]; Output nodes: [%d]%n%n", weightDimensions(0)[1],
                        weightDimensions(weights.length - 1)[0]));

        for (int i = 0; i < weights.length; i++) {
            int[] dims = weightDimensions(i);
            b.append(String.format("\t\tLayer %d : [%d X %d]%n", i, dims[0], dims[1]))
                    .append(String.format("\t\tActivation function from this layer: %s", functions[i].getName())).append("\n");
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
    public static NeuralNetwork readObject(String path) throws IOException {
        NeuralNetwork network = null;
        File file;
        path = (path.endsWith(".ser") ? path : path + ".ser");

        try (FileInputStream fs = new FileInputStream(file = new File(path));
             ObjectInputStream os = new ObjectInputStream(fs)) {

            network = (NeuralNetwork) os.readObject();

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
    public static NeuralNetwork readObject(final File file) throws IOException {
        NeuralNetwork neuralNetwork = null;
        try (FileInputStream fs = new FileInputStream(file); ObjectInputStream stream = new ObjectInputStream(fs)) {
            neuralNetwork = (NeuralNetwork) stream.readObject();

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

}
