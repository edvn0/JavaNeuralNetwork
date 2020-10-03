package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.NeuralNetwork.BackPropContainer;
import neuralnetwork.inputs.NetworkInput;

public class BackpropagationCallable implements Callable<NeuralNetwork.BackPropContainer> {

    private NeuralNetwork n;
    private List<List<NetworkInput>> threadSplits;

    public BackpropagationCallable(int batchSize, NeuralNetwork clone, List<NetworkInput> splitBatch) {
        this.n = clone;
        this.threadSplits = getSplitBatch(splitBatch, batchSize);
    }

    private List<List<NetworkInput>> getSplitBatch(final List<NetworkInput> list, final int splitSize) {
        int partitionSize = splitSize;
        List<List<NetworkInput>> partitions = new ArrayList<>();

        for (int i = 0; i < list.size(); i += partitionSize) {
            partitions.add(list.subList(i, Math.min(i + partitionSize, list.size())));
        }
        return partitions;
    }

    @Override
    public BackPropContainer call() throws Exception {
        final int size = threadSplits.get(0).size() * threadSplits.size();
        double loss = 1d;// / (size);
        final List<OjAlgoMatrix> deltaWeights = n.getdW();
        final List<OjAlgoMatrix> deltaBiases = n.getdB();
        threadSplits.forEach(e -> {
            this.n.evaluateTrainingExample(e);
            for (int i = 0; i < deltaBiases.size(); i++) {
                deltaWeights.set(i, deltaWeights.get(i).add(n.getSingleDw(i).multiply(loss)));
                deltaBiases.set(i, deltaBiases.get(i).add(n.getSingleDb(i).multiply(loss)));
            }
            this.n.learnFromDeltas();
        });

        return new BackPropContainer(deltaWeights, deltaBiases);

    }

}
