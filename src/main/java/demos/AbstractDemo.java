package demos;

import java.util.List;

import org.apache.log4j.BasicConfigurator;

import lombok.extern.slf4j.Slf4j;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.inputs.NetworkInput;
import utilities.types.Pair;
import utilities.types.Triple;

@Slf4j
public abstract class AbstractDemo<M> {

    void demo() {
        BasicConfigurator.configure();
        final NeuralNetwork<M> network = createNetwork();
        final Triple<List<NetworkInput<M>>, List<NetworkInput<M>>, List<NetworkInput<M>>> trainValidateTest = getData();
        final Pair<Integer, Integer> epochBatch = epochBatch();
        final TrainingMethod trainingMethod = networkTrainingMethod();
        final String outputPath = outputDirectory();

        network.display();

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

        double confusion = network.testEvaluation(trainValidateTest.getRight(), 50);
        double loss = network.testLoss(trainValidateTest.getRight());
        log.info("\nCorrectly evaluated {}% of the test set.\nFinal loss: {}", confusion * 100, loss);

    }

    protected abstract String outputDirectory();

    protected abstract TrainingMethod networkTrainingMethod();

    protected abstract Pair<Integer, Integer> epochBatch();

    protected abstract Triple<List<NetworkInput<M>>, List<NetworkInput<M>>, List<NetworkInput<M>>> getData();

    protected abstract NeuralNetwork<M> createNetwork();

    public enum TrainingMethod {
        VERBOSE, METRICS, NORMAL
    }

}
