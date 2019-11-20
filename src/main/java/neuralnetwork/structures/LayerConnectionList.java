package neuralnetwork.structures;

import math.ActivationFunction;
import math.TanhFunction;
import matrix.Matrix;

public class LayerConnectionList {

	private int totalLayers;

	// 0 based connections, i.e., connection 0 is from Layer 0 to Layer 1.
	private Matrix[] weights;

	// 0 based layering, i.e. index 0 in layers is layer 0.
	private Layer[] layers;

	// Functions mapped to {@link Layer} indexInNetwork
	private ActivationFunction[] functions;

	public LayerConnectionList(ActivationFunction[] functions, int... sizes) {
		this.functions = functions;
		this.totalLayers = sizes.length;

		weights = new Matrix[totalLayers - 1];
		layers = new Layer[totalLayers - 1];

		createLayers(sizes);
		initialiseWeights(sizes);
	}

	public LayerConnectionList(double[][] activatedLayers, Matrix[] weights,
		Matrix[] biases) {
		this.weights = new Matrix[weights.length];
		this.layers = new Layer[activatedLayers.length];

		layers[0] = new Layer(biases[0].getRows(), biases[0], 0, false, activatedLayers[0],
			new TanhFunction());
		for (int i = 1; i < layers.length; i++) {
			double[] val = activatedLayers[i];
			layers[i] = new Layer(biases[i - 1].getRows(), biases[i - 1], i,
				true, val, new TanhFunction());
		}

		int k = 0;
		for (Matrix m : weights) {
			this.weights[k] = m;
			k++;
		}
	}

	private void initialiseWeights(int[] sizes) {
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.weights[i] = Matrix.random(sizes[i + 1], sizes[i]);
		}
	}


	private void createLayers(int[] sizes) {
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.layers[i] = new Layer(sizes[i + 1], Matrix.random(sizes[i + 1], 1), functions[i]);
		}
	}

	public int getTotalLayers() {
		return this.totalLayers;
	}

	public Layer getLayer(int i) {
		return layers[i];
	}

	/**
	 * Set the matrix (weight)
	 */
	public void setWeight(int i, Matrix newWeights) {
		this.weights[i].setData(newWeights);
	}

	public Matrix[] getWeights() {
		return this.weights;
	}

	public Matrix getWeight(int i) {
		return this.weights[i];
	}

	public void setLayerBias(int i, Matrix outputMatrix) {
		this.layers[i].setBias(outputMatrix);
	}

	public Matrix[] getBiases() {
		Matrix[] biases = new Matrix[getTotalLayers()];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			biases[i] = getLayer(i).getBias();
		}
		return biases;
	}

	public void printWeights() {
		for (Matrix w : weights) {
			System.out.println(w);
			System.out.println();
		}
	}

	public void printLayers() {
		for (Layer l : layers) {
			System.out.println(l.getValues());
		}
	}

	public Layer[] getLayers() {
		return this.layers;
	}

	public void resetLayers() {
		for (Layer l : layers) {
			l.reset();
		}
	}

	public Matrix getBias(int i) {
		return this.layers[i].getBias();
	}
}
