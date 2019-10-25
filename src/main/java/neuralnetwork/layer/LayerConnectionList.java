package neuralnetwork.layer;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;
import matrix.Matrix;

public class LayerConnectionList {

	private HashMap<Connection, Matrix> weightMatrixMap;

	private int inputNodes, hiddenLayers, nodesInHidden, outputNodes;
	Connection[] connections;
	Layer[] layers;

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
		layers[0] = new Layer(this.inputNodes, Matrix.random(this.inputNodes, 1), 0, false);
		for (int i = 0; i < hiddenLayers; i++) {
			int index = i + 1;
			layers[index] = new Layer(this.nodesInHidden, Matrix.random(this.nodesInHidden, 1),
				(index), true);
		}
		int len = this.layers.length - 1;
		layers[len] = new Layer(this.outputNodes,
			Matrix.random(this.outputNodes, 1),
			len, true);
	}

	private void createConnections() {
		for (int i = 0; i < this.layers.length - 1; i++) {
			connections[i] = new Connection(this.layers[i], this.layers[i + 1]);
		}
	}

	private void initialiseWeights() {
		for (int i = 0; i < this.connections.length; i++) {
			int rows = this.connections[i].getTo().getNodesInLayer();
			int cols = this.connections[i].getFrom().getNodesInLayer();
			Matrix weights = Matrix.random(rows, cols);
			this.weightMatrixMap.put(this.connections[i], weights);
		}
	}

	public Matrix calculateLayer(Matrix inputs, int connection, int layerIndex) {

		// Get weights from map, get layer to retrieve bias from (if it has bias)
		Matrix weights = getWeights(connection);
		Layer layer = getLayer(layerIndex);
		boolean hasBias = layer.hasBias();

		if (inputs.getRows() != weights.getColumns()) {
			throw new IllegalArgumentException(
				"Columns and rows do not match, input had " + inputs.getRows()
					+ " and the weights had: " + weights.getColumns());
		}

		Matrix output = weights.multiply(inputs);

		if (hasBias) {
			output.add(layer.getBias());
		}

		return output;
	}


	public Set<Entry<Connection, Matrix>> getWeightsEntries() {
		return this.weightMatrixMap.entrySet();
	}

	public int amountWeightMatrices() {
		return this.weightMatrixMap.size();
	}

	public Matrix getWeights(int i) {
		return this.weightMatrixMap.get(connections[i]);
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

	public void setWeights(int i, Matrix newWeights) {
		this.weightMatrixMap.get(connections[i]).setData(newWeights);
	}
}
