package neuralnetwork.layer;

import matrix.Matrix;

public class Layer {

	private int nodes;
	private Matrix bias;

	private double[] values;

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
		values = new double[nodes];
	}

	Layer(int nodes, int indexInNetwork) {
		this(nodes, null, indexInNetwork, false);
		values = new double[nodes];
	}

	void setValue(double val, int index) {
		this.values[index] = val;
	}

	void setValue(Matrix values) {
		this.values = values.toArray();
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

	public Matrix getValues() {
		double[][] newValues = new double[this.values.length][1];
		int k = 0;
		for (double d : this.values) {
			newValues[k++][0] = d;
		}

		return new Matrix(newValues);
	}

	public void setBias(Matrix outputMatrix) {
		this.bias = outputMatrix;
	}
}
