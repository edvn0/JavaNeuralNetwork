package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import math.linearalgebra.Matrix;
import neuralnetwork.NeuralNetwork.BackPropContainer;
import neuralnetwork.inputs.NetworkInput;

public class BackpropagationCallable<M> implements Callable<NeuralNetwork.BackPropContainer<M>> {

    private NeuralNetwork<M> n;
    private List<List<NetworkInput<M>>> threadSplits;

    public BackpropagationCallable(int batchSize, NeuralNetwork<M> clone, List<NetworkInput<M>> splitBatch) {
        this.n = clone;
        this.threadSplits = getSplitBatch(splitBatch, batchSize);
    }

    private List<List<NetworkInput<M>>> getSplitBatch(final List<NetworkInput<M>> list, final int splitSize) {
        int partitionSize = splitSize;
        List<List<NetworkInput<M>>> partitions = new ArrayList<>();

        for (int i = 0; i < list.size(); i += partitionSize) {
            partitions.add(list.subList(i, Math.min(i + partitionSize, list.size())));
        }
        return partitions;
    }

    @Override
    public BackPropContainer<M> call() throws Exception {
        final int size = threadSplits.get(0).size() * threadSplits.size();
        double loss = 1d;// / (size);
        final List<Matrix<M>> deltaWeights = n.getdW();
        final List<Matrix<M>> deltaBiases = n.getdB();
        threadSplits.forEach(e -> {
            this.n.evaluateTrainingExample(e);
            for (int i = 0; i < deltaBiases.size(); i++) {
                deltaWeights.set(i, deltaWeights.get(i).add(n.getSingleDw(i).multiply(loss)));
                deltaBiases.set(i, deltaBiases.get(i).add(n.getSingleDb(i).multiply(loss)));
            }
            this.n.learnFromDeltas();
        });

        return new BackPropContainer<>(deltaWeights, deltaBiases);

    }

}
