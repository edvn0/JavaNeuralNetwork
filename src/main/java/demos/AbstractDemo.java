package demos;

import java.util.List;

import org.apache.log4j.BasicConfigurator;

import lombok.extern.slf4j.Slf4j;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.inputs.NetworkInput;
import utilities.types.Pair;
import utilities.types.Triple;

@Slf4j
public abstract class AbstractDemo {

    void demo() {
        BasicConfigurator.configure();
        final NeuralNetwork network = createNetwork();
        final Triple<List<NetworkInput>, List<NetworkInput>, List<NetworkInput>> trainValidateTest = getData();
        final Pair<Integer, Integer> epochBatch = epochBatch();
        final TrainingMethod trainingMethod = networkTrainingMethod();
        final String outputPath = outputDirectory();

        network.display();

        long t1 = System.nanoTime();

        switch (trainingMethod) {
            case METRICS:
                network.trainWithMetrics(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right(), outputPath);
                break;
            case NORMAL:
                network.train(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right());
                break;
            case VERBOSE:
                network.trainVerbose(trainValidateTest.getLeft(), trainValidateTest.getMiddle(), epochBatch.left(),
                        epochBatch.right());
                break;
            default:
                break;

        }

        long t2 = System.nanoTime();

        double confusion = network.testEvaluation(trainValidateTest.getRight(), 50);
        double loss = network.testLoss(trainValidateTest.getRight());
        log.info("\nCorrectly evaluated {}% of the test set.\nFinal loss: {}", confusion * 100, loss);
        log.info("\nTotal time taken for training: {}.", (t2-t1)*1e-6);

    }

    protected abstract String outputDirectory();

    protected abstract TrainingMethod networkTrainingMethod();

    protected abstract Pair<Integer, Integer> epochBatch();

    protected abstract Triple<List<NetworkInput>, List<NetworkInput>, List<NetworkInput>> getData();

    protected abstract NeuralNetwork createNetwork();

    public enum TrainingMethod {
        VERBOSE, METRICS, NORMAL
    }

}
