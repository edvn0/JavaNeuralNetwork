package utilities.serialise.serialisers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.List;
import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;

public class OjAlgoLayeredSerializer {

	private final Gson gson;
	private Type network = new TypeToken<LayeredNeuralNetwork<Primitive64Matrix>>() {
	}.getType();

	//	private final ActivationFunction<M> activationFunction;
	//	private final int neurons;
	//
	//	private Matrix<M> weight;
	//	private Matrix<M> bias;
	//
	//	private NetworkLayer<M> previousLayer;

	public OjAlgoLayeredSerializer() {

		GsonBuilder gsonb = new GsonBuilder();

		Type activationFunction = new TypeToken<ActivationFunction<Primitive64Matrix>>() {
		}.getType();
		Type optimiser = new TypeToken<Optimizer<Primitive64Matrix>>() {
		}.getType();
		Type evalFunction = new TypeToken<EvaluationFunction<Primitive64Matrix>>() {
		}.getType();
		Type costFunction = new TypeToken<CostFunction<Primitive64Matrix>>() {
		}.getType();
		Type matrices = new TypeToken<Matrix<Primitive64Matrix>>() {
		}.getType();
		Type layers = new TypeToken<List<NetworkLayer<Primitive64Matrix>>>() {
		}.getType();

		Type layer = new TypeToken<NetworkLayer<Primitive64Matrix>>() {
		}.getType();

		gsonb.registerTypeAdapter(activationFunction,
			(JsonSerializer<ActivationFunction<Primitive64Matrix>>) (src, typeOfSrc, context) -> new JsonPrimitive(
				src.getName()));

		gsonb.registerTypeAdapter(optimiser,
			(JsonSerializer<Optimizer<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonObject inner = new JsonObject();
				var info = src.params();

				if (info != null) {
					int i = 1;
					for (var e : info.values()) {
						inner.addProperty("v" + i, e);
						i++;
					}
				}
				inner.addProperty("name", src.name());

				return inner;
			});

		gsonb.registerTypeAdapter(evalFunction,
			(JsonSerializer<EvaluationFunction<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonObject inner = new JsonObject();
				var info = src.params();

				inner.addProperty("name", src.name());
				if (info != null) {
					info.forEach((a, b) -> inner.addProperty("value", b));
				}

				return inner;
			});

		gsonb.registerTypeAdapter(costFunction,
			(JsonSerializer<CostFunction<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonObject name = new JsonObject();
				name.addProperty("name", src.name());
				return name;
			});

		gsonb.registerTypeAdapter(matrices,
			(JsonSerializer<Matrix<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonArray outer = new JsonArray();
				double[][] out = src.rawCopy();
				for (double[] ds : out) {
					JsonArray inner = new JsonArray();
					for (double d : ds) {
						inner.add(d);
					}
					outer.add(inner);
				}

				return outer;
			});

		gsonb.registerTypeAdapter(layer,
			(JsonSerializer<NetworkLayer<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonObject jsonlayer = new JsonObject();
				jsonlayer
					.add("function",
						context.serialize(src.getFunction(), activationFunction));

				jsonlayer.add("neurons", context.serialize(src.getNeurons()));

				jsonlayer.add("weight",
					context.serialize(src.getWeight(), matrices));

				jsonlayer
					.add("bias", context.serialize(src.getBias(), matrices));
				return jsonlayer;
			});

		gsonb.registerTypeAdapter(layers,
			(JsonSerializer<List<NetworkLayer<Primitive64Matrix>>>) (src, typeOfSrc, context) -> {
				JsonArray arrayOfLayers = new JsonArray();
				int i = 0;
				for (var l : src) {
					JsonObject lA = new JsonObject();

					lA.add("layer", context.serialize(l, layer));
					lA.add("index", new JsonPrimitive(i++));

					arrayOfLayers.add(lA);
				}

				return arrayOfLayers;
			});

		gsonb.registerTypeAdapter(network,
			(JsonSerializer<LayeredNeuralNetwork<Primitive64Matrix>>) (src, typeOfSrc, context) -> {
				JsonObject network = new JsonObject();
				network.add("layers",
					context.serialize(src.getLayers(), layers));

				network.add("costFunction", context.serialize(src.getCostFunction(), costFunction));
				network.add("evaluationFunction",
					context.serialize(src.getEvaluationFunction(), evalFunction));
				network.add("optimizer", context.serialize(src.getOptimizer(), optimiser));

				return network;
			});

		this.gson = gsonb.create();
	}

	public void serialise(File f, LayeredNeuralNetwork<Primitive64Matrix> ojAlgoNetwork) {
		String json = serialiseToString(ojAlgoNetwork);

		try (FileWriter fw = new FileWriter(f, false)) {
			fw.write(json);
		} catch (IOException e) {

		}
	}

	public String serialiseToString(LayeredNeuralNetwork<Primitive64Matrix> ojAlgoNetwork) {
		return gson.toJson(ojAlgoNetwork, network);
	}
}
