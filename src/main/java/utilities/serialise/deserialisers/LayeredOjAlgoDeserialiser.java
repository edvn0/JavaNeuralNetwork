package utilities.serialise.deserialisers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.optimizers.Optimizer;
import neuralnetwork.LayeredNeuralNetwork;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.serialise.ConverterUtil;
import utilities.serialise.adapters.LayeredOjAlgoNetworkDeserializer;

public class LayeredOjAlgoDeserialiser {

	private Gson gson;
	private Type network = new TypeToken<LayeredNeuralNetwork<Primitive64Matrix>>() {
	}.getType();

	public LayeredOjAlgoDeserialiser() {
		GsonBuilder gsonb = new GsonBuilder();

		Type activationFunction = new TypeToken<ActivationFunction<Primitive64Matrix>>() {
		}.getType();
		Type optimiser = new TypeToken<Optimizer<Primitive64Matrix>>() {
		}.getType();
		Type evalFunction = new TypeToken<EvaluationFunction<Primitive64Matrix>>() {
		}.getType();
		Type costFunction = new TypeToken<CostFunction<Primitive64Matrix>>() {
		}.getType();
		Type matrix = new TypeToken<Matrix<Primitive64Matrix>>() {
		}.getType();
		Type layers = new TypeToken<List<NetworkLayer<Primitive64Matrix>>>() {
		}.getType();

		Type layer = new TypeToken<NetworkLayer<Primitive64Matrix>>() {
		}.getType();

		gsonb.registerTypeAdapter(activationFunction,
			(JsonDeserializer<ActivationFunction<Primitive64Matrix>>) (json, typeOfT, context) -> {

				return ConverterUtil.ojFunctions.get(json.getAsString());
			});

		gsonb.registerTypeAdapter(optimiser,
			(JsonDeserializer<Optimizer<Primitive64Matrix>>) (json, typeOfT, context) -> {
				JsonObject obj = json.getAsJsonObject();

				Optimizer<Primitive64Matrix> op = ConverterUtil.ojOptimisers
					.get(obj.get("name").getAsString());

				double lR = tryToFind(obj, "v1"); // always learning rate
				double v2 = tryToFind(obj, "v2"); // beta1 or momentum
				double v3 = tryToFind(obj, "v3"); // beta2

				double[] vals = new double[]{lR, v2, v3};

				op.init(vals);

				return op;
			});

		gsonb.registerTypeAdapter(costFunction,
			(JsonDeserializer<CostFunction<Primitive64Matrix>>) (json, typeOfT, context) -> {
				JsonObject obj = json.getAsJsonObject();

				CostFunction<Primitive64Matrix> cf = ConverterUtil.ojCostFunctions
					.get(obj.get("name").getAsString());

				return cf;
			});

		gsonb.registerTypeAdapter(evalFunction,
			(JsonDeserializer<EvaluationFunction<Primitive64Matrix>>) (json, typeOfT, context) -> {
				JsonObject obj = json.getAsJsonObject();

				EvaluationFunction<Primitive64Matrix> cf = ConverterUtil.ojEvaluators
					.get(obj.get("name").getAsString());

				double val = tryToFind(obj, "v1");

				cf.init(val);

				return cf;
			});

		gsonb.registerTypeAdapter(matrix,
			(JsonDeserializer<Matrix<Primitive64Matrix>>) (json, typeOfT, context) -> {
				JsonArray matrix1 = json.getAsJsonArray();
				int rows = matrix1.size();
				int cols = matrix1.get(0).getAsJsonArray().size();

				double[][] values = new double[rows][cols];
				for (int k = 0; k < rows; k++) {
					JsonArray row = matrix1.get(k).getAsJsonArray();
					for (int j = 0; j < cols; j++) {
						values[k][j] = row.get(j).getAsDouble();
					}
				}

				return new OjAlgoMatrix(values);
			});

		gsonb.registerTypeAdapter(layer,
			(JsonDeserializer<NetworkLayer<Primitive64Matrix>>) (src, typeOfSrc, context) -> {

				var data = src.getAsJsonObject().get("layer_data").getAsJsonObject();
				ActivationFunction<Primitive64Matrix> function = context
					.deserialize(data.get("function"), activationFunction);

				int neurons = context.deserialize(data.get("neurons"), int.class);

				NetworkLayer<Primitive64Matrix> out = new NetworkLayer<Primitive64Matrix>(function,
					neurons);
				Matrix<Primitive64Matrix> weight, bias;

				weight = context.deserialize(data.get("weight"), matrix);
				bias = context.deserialize(data.get("bias"), matrix);

				out.setBias(bias);
				out.setWeight(weight);
				return out;
			});

		gsonb.registerTypeAdapter(layers,
			(JsonDeserializer<List<NetworkLayer<Primitive64Matrix>>>) (src, typeOfSrc, context) -> {
				JsonArray arrayOfLayers = src.getAsJsonArray();
				List<NetworkLayer<Primitive64Matrix>> layerList = new ArrayList<>();
				layerList.add(context.deserialize(arrayOfLayers.get(0).getAsJsonObject(), layer));
				var temp = context.<NetworkLayer<Primitive64Matrix>>deserialize(
					arrayOfLayers.get(0).getAsJsonObject(), layer);
				for (int i = 1; i < arrayOfLayers.size(); i++) {
					var tempLayer = context
						.<NetworkLayer<Primitive64Matrix>>deserialize(
							arrayOfLayers.get(i).getAsJsonObject(), layer);

					tempLayer.setPrecedingLayer(temp);
					layerList.add(tempLayer);
					temp = tempLayer;
				}

				return layerList;
			});

		gsonb.registerTypeAdapter(network, new LayeredOjAlgoNetworkDeserializer());

		this.gson = gsonb.create();
	}

	public LayeredNeuralNetwork<Primitive64Matrix> deserialise(File jsonFile) {

		LayeredNeuralNetwork<Primitive64Matrix> out = null;
		try (JsonReader reader = new JsonReader(new FileReader(jsonFile))) {
			out = gson.fromJson(reader, network);
		} catch (IOException | JsonSyntaxException e) {
			e.printStackTrace();
		}

		return out;
	}

	public NeuralNetwork<Primitive64Matrix> deserialise(String json) {
		json = json.trim();
		return gson.fromJson(json, network);
	}

	private double tryToFind(JsonObject obj, String el) {
		double val = 0;
		try {
			val = obj.get(el).getAsDouble();
		} catch (NullPointerException | ClassCastException | IllegalStateException e) {
			val = 0;
		}
		return val;
	}

}
