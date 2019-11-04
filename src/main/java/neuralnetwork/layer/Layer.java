package neuralnetwork.layer;

import matrix.Matrix;

public class Layer {

	private int nodes;
	private Matrix bias;

	private boolean hasBias;

	private int indexInNetwork;

	Layer() {
		nodes = 0;
		bias = null;
		indexInNetwork = -1;
	}

	Layer(int nodes, Matrix bias, int indexInNetwork, boolean hiddenLayer) {
		this.nodes = nodes;
		this.bias = bias;
		this.indexInNetwork = indexInNetwork;
		this.hasBias = hiddenLayer;
	}

	void calculateBias(Matrix old) {
		this.bias.add(old);
	}

	public int getNodes() {
		return nodes;
	}

	public Matrix getBias() {
		return bias;
	}

	public int getIndexInNetwork() {
		return indexInNetwork;
	}

	public int getNodesInLayer() {
		return this.nodes;
	}

	@Override
	public String toString() {
		return "[ " + this.nodes + " ]";
	}

	boolean hasBias() {
		return this.hasBias;
	}
}
