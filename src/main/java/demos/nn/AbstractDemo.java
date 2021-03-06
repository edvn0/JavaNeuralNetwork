package demos.nn;

import java.util.List;
import lombok.extern.slf4j.Slf4j;
import neuralnetwork.DeepLearnable;
import neuralnetwork.inputs.NetworkInput;
import org.apache.log4j.BasicConfigurator;
import utilities.types.Pair;
import utilities.types.Triple;

@Slf4j
public abstract class AbstractDemo<M> {

	protected void demo() {
		BasicConfigurator.configure();
		final DeepLearnable<M> network = createNetwork();
		final Triple<List<NetworkInput<M>>, List<NetworkInput<M>>, List<NetworkInput<M>>> trainValidateTest = getData();
		final Pair<Integer, Integer> epochBatch = epochBatch();
		final TrainingMethod trainingMethod = networkTrainingMethod();
		final String outputPath = outputDirectory();

		network.display();

		long t1 = System.nanoTime();

		switch (trainingMethod) {
			case METRICS:
				network.trainWithMetrics(trainValidateTest.getLeft(), trainValidateTest.getMiddle(),
					epochBatch.left(),
					epochBatch.right(), outputPath);
				break;
			case NORMAL:
				network.train(trainValidateTest.getLeft(), trainValidateTest.getMiddle(),
					epochBatch.left(),
					epochBatch.right());
				break;
			default:
				break;

		}

		long t2 = System.nanoTime();

		double confusion = network.testEvaluation(trainValidateTest.getRight(), 50);
		double loss = network.testLoss(trainValidateTest.getRight());
		log.info("\nCorrectly evaluated {}% of the test set.\nFinal loss: {}", confusion * 100,
			loss);
		log.info("\nTotal time taken for training: {} seconds.", ((t2 - t1) * 1e-9));

		serialise(network);
		log.info("Serialised the network.");
	}

	protected abstract void serialise(DeepLearnable<M> in);

	/**
	 * Where should we output network information?
	 *
	 * @return path to output directory.
	 */
	protected abstract String outputDirectory();

	/**
	 * What method of training should be chosen?
	 *
	 * @return training method.
	 */
	protected abstract TrainingMethod networkTrainingMethod();

	/**
	 * @return A tuple of (epochs, batchSize)
	 */
	protected abstract Pair<Integer, Integer> epochBatch();

	/**
	 * Generate and split data into training, validation and testing.
	 *
	 * @return (training, validation, testing) tuple.
	 */
	protected abstract Triple<List<NetworkInput<M>>, List<NetworkInput<M>>, List<NetworkInput<M>>> getData();

	/**
	 * Construct your network to train with.
	 *
	 * @return the network.
	 */
	protected abstract DeepLearnable<M> createNetwork();

	public enum TrainingMethod {
		METRICS, NORMAL
	}

}
