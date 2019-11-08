package neuralnetwork.layer;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;
import matrix.Matrix;

public class LayerConnectionList {

	private HashMap<Connection, Matrix> weightMatrixMap;

	private int inputNodes, hiddenLayers, nodesInHidden, outputNodes;
	private Connection[] connections;
	private Layer[] layers;

	public LayerConnectionList() {
		weightMatrixMap = new HashMap<>();
	}

	public LayerConnectionList(int inputNodes, int hiddenLayers, int nodesInHidden,
		int outputNodes) {
		this.inputNodes = inputNodes;
		this.hiddenLayers = hiddenLayers;
		this.nodesInHidden = nodesInHidden;
		this.outputNodes = outputNodes;

		weightMatrixMap = new HashMap<>();
		connections = new Connection[(2 + hiddenLayers) - 1];
		layers = new Layer[2 + hiddenLayers];

		createLayers();
		createConnections();

		initialiseWeights();
	}

	private void createLayers() {
		layers[0] = new Layer(this.inputNodes, 0);
		for (int i = 0; i < hiddenLayers; i++) {
			int index = i + 1;
			layers[index] = new Layer(this.nodesInHidden, Matrix.random(this.nodesInHidden, 1),
				(index), true);
		}
		int len = this.layers.length - 1;
		layers[len] = new Layer(this.outputNodes, Matrix.random(this.outputNodes, 1), len, true);
	}


	private void createConnections() {
		for (int i = 0; i < this.layers.length - 1; i++) {
			Layer from = this.layers[i];
			Layer to = this.layers[i + 1];
			connections[i] = new Connection(from, to,
				Matrix.random(to.getNodesInLayer(), from.getNodesInLayer()));
		}
	}

	private void initialiseWeights() {
		for (Connection connection : this.connections) {
			int rows = connection.getTo().getNodesInLayer();
			int cols = connection.getFrom().getNodesInLayer();
			Matrix weights = Matrix.random(rows, cols);
			this.weightMatrixMap.put(connection, weights);
		}
	}

	/**
	 * Inputs, what to calculate, index:what connection
	 */
	public Matrix calculateLayer(Matrix inputs, int index) {

		if (inputs == null) {
			return null;
		}

		// Get weights from map, get layer to retrieve bias from (if it has bias)

		Matrix weights = getWeight(index);
		Layer layer = getLayer(index + 1);
		boolean hasBias = layer.hasBias();

		if (inputs.getRows() != weights.getColumns()) {
			throw new IllegalArgumentException(
				"Columns and rows do not match, input had " + inputs.getRows()
					+ " and the weights had: " + weights.getColumns());
		}

		Matrix layerOutput = weights.multiply(inputs);

		if (hasBias) {
			layerOutput = layerOutput.add(layer.getBias());
		}

		return layerOutput;
	}


	public Set<Entry<Connection, Matrix>> getWeightsEntries() {
		return this.weightMatrixMap.entrySet();
	}

	public int amountOfWeights() {
		return this.weightMatrixMap.size();
	}

	public Layer getLayer(int i) {
		return layers[i];
	}

	public int getOutputNodes() {
		return this.outputNodes;
	}

	public int getInputNodes() {
		return this.inputNodes;
	}

	public void setWeight(int i, Matrix newWeights) {
		this.weightMatrixMap.get(connections[i]).setData(newWeights);
	}

	public Matrix[] getWeights() {
		Matrix[] returns = new Matrix[this.amountOfWeights()];
		for (int i = 0; i < this.amountOfWeights(); i++) {
			returns[i] = this.weightMatrixMap.get(connections[i]);
		}
		return returns;
	}

	public Matrix getWeight(int i) {
		return this.weightMatrixMap.get(connections[i]);
	}

	public void setActiveLayer(int i, Matrix outputMatrix) {
		this.layers[i].setValue(outputMatrix);
	}

	public void setActiveLayerBias(int i, Matrix outputMatrix) {
		this.layers[i].setBias(outputMatrix);
	}

	public Matrix getActiveLayer(int i) {
		return this.layers[i].getValues();
	}

	public Matrix[] getLayers() {
		Matrix[] layers = new Matrix[this.layers.length];
		for (int i = 0; i < layers.length; i++) {
			layers[i] = getLayer(i).getValues();
		}
		return layers;
	}

	public Matrix[] getBiases() {
		Matrix[] layers = new Matrix[this.layers.length];
		for (int i = 0; i < layers.length; i++) {
			layers[i] = getLayer(i).getBias();
		}
		return layers;
	}

	public void printWeights() {
		for (Connection w : connections) {
			System.out
				.println(w.getFrom().getIndexInNetwork() + " " + w.getTo().getIndexInNetwork());
			w.getWeights().show();
			System.out.println();
		}
	}

	public void printLayers() {
		for (Layer l : layers) {
			l.getValues().show();
		}
	}
}
