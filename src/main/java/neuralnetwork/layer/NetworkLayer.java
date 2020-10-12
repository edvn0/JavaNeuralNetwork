package neuralnetwork.layer;

import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicInteger;
import math.activations.ActivationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;

public class NetworkLayer<M> {

	private final ActivationFunction<M> activationFunction;
	private final int neurons;

	// Represents the data after activating this layer
	private final ThreadLocal<ZVector<M>> activated;
	private Matrix<M> weight;
	private Matrix<M> bias;

	private NetworkLayer<M> previousLayer;

	private final transient AtomicInteger deltasAdded;
	private transient Matrix<M> deltaWeight;
	private transient Matrix<M> deltaBias;

	public NetworkLayer(ActivationFunction<M> activationFunction, int neurons) {
		this.activationFunction = activationFunction;
		this.activated = new ThreadLocal<>();
		this.neurons = neurons;
		this.deltasAdded = new AtomicInteger(0);
	}

	public NetworkLayer(NetworkLayer<M> in) {
		this(in.activationFunction, in.neurons);
		this.weight = in.weight;
		this.bias = in.bias;
		this.previousLayer = in.previousLayer;
	}

	public ZVector<M> calculate(ZVector<M> in) {
		if (!hasPrecedingLayer()) {
			var out = new ZVector<M>(in);
			this.activated.set(out);
		} else {
			var out = new ZVector<>(
				activationFunction.function(this.weight.multiply(in.getZVector()).add(bias)));
			this.activated.set(out);
		}

		return this.activated.get();
	}

	public synchronized void addDeltas(Matrix<M> deltaWeights, Matrix<M> deltaBias) {
		this.deltaWeight = this.deltaWeight.add(deltaWeights);
		this.deltaBias = this.deltaBias.add(deltaBias);
		this.deltasAdded.incrementAndGet();
	}

	public synchronized void fit(int index, Optimizer<M> optimizer) {
		int added = this.deltasAdded.get();
		if (added > 0) {
			var averageDeltaW = this.deltaWeight.mapElements(e -> e / added);
			var averageDeltaB = this.deltaBias.mapElements(e -> e / added);
			optimizer.changeBias(index, this, averageDeltaB);
			optimizer.changeWeight(index, this, averageDeltaW);

			this.deltaWeight.mapElementsMutable(e -> 0d);
			this.deltaBias.mapElementsMutable(e -> 0d);

			this.deltasAdded.set(0);
		}
	}

	public int getNeurons() {
		return neurons;
	}

	public Matrix<M> getBias() {
		return this.bias;
	}

	public void setBias(Matrix<M> newBias) {
		this.bias = newBias;
	}

	public Matrix<M> getWeight() {
		return this.weight;
	}

	public void setWeight(Matrix<M> newWeights) {
		this.weight = newWeights;
	}

	public Matrix<M> activation() {
		return this.activated.get().getZVector();
	}

	public boolean hasPrecedingLayer() {
		return this.previousLayer != null;
	}

	public ActivationFunction<M> getFunction() {
		return this.activationFunction;
	}

	public NetworkLayer<M> precedingLayer() {
		return this.previousLayer;
	}

	public void setPrecedingLayer(NetworkLayer<M> prev) {
		this.previousLayer = prev;
	}


	@Override
	public String toString() {

		if (this.weight == null) {
			return new StringJoiner(", ", NetworkLayer.class.getSimpleName() + "[", "]")
				.add("activationFunction=" + activationFunction.getName()).add("neurons=" + neurons)
				.toString();
		}

		return new StringJoiner(", ", NetworkLayer.class.getSimpleName() + "[", "]")
			.add("activationFunction=" + activationFunction.getName())
			.add("weight=[" + weight.rows()).add(weight.cols() + "]")
			.add("bias=[" + bias.rows()).add(bias.cols() + "]")
			.add("neurons=" + neurons)
			.toString();
	}

	public void setDeltaWeight(final Matrix<M> layerDeltaWeight) {
		this.deltaWeight = layerDeltaWeight;
	}

	public void setDeltaBias(final Matrix<M> layerDeltaWeight) {
		this.deltaBias = layerDeltaWeight;
	}
}
