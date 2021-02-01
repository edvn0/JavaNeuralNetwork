package utilities.serialise.deserialisers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Optional;
import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.ParameterInitializer;
import neuralnetwork.initialiser.SimpleInitializer;
import neuralnetwork.layer.LayeredNetworkBuilder;
import neuralnetwork.layer.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import utilities.serialise.NetworkDataCache;

public class SimpleDeserializer {

	private static final Type networkType = new TypeToken<LayeredNeuralNetwork<SMatrix>>() {
	}.getType();
	private static final JsonDeserializer<LayeredNeuralNetwork<SMatrix>> deserializer = (src, type, context) -> {
		// "layers" ("neurons") or ("neurons", "activation", "weight", "bias")
		// "optimizer" ("name") or ("name", "params")
		// "initializer" ("name", "weightmethod", "biasmethod")
		// "evaluator" ("name") or ("name", "params")
		// "costfunction" ("name")
		// "clipping"

		var network = src.getAsJsonObject();
		var layers = network.get("layers").getAsJsonArray();
		var firstLayer = layers.remove(0);
		LayeredNetworkBuilder<SMatrix> nBuilder = new LayeredNetworkBuilder<SMatrix>();
		NetworkLayer<SMatrix> first = new NetworkLayer<>(new DoNothingFunction<SMatrix>(),
			firstLayer.getAsJsonObject().get("neurons").getAsInt());

		var initObj = network.get("initializer").getAsJsonObject();
		InitialisationMethod wM, bM;
		wM = InitialisationMethod.get(initObj.get("weightmethod").getAsString());
		bM = InitialisationMethod.get(initObj.get("biasmethod").getAsString());
		SimpleInitializer init = (SimpleInitializer) ParameterInitializer
			.get(wM, bM, initObj.get("name").getAsString(),
				SMatrix.class);

		var arr = network.get("networkLayout").getAsJsonArray();
		int[] sizes = new int[arr.size()];
		int layoutIndex = 0;
		for (var el : arr) {
			sizes[layoutIndex++] = el.getAsInt();
		}

		init.init(sizes);

		var deltaB = init.getDeltaBiasParameters();
		var deltaW = init.getDeltaWeightParameters();

		nBuilder.layer(first);

		int layerIndex = 1;
		for (var l : layers) {
			var lSrc = l.getAsJsonObject();
			int neurons = lSrc.get("neurons").getAsInt();
			double l2 = lSrc.get("l2").getAsDouble();
			ActivationFunction<SMatrix> act = NetworkDataCache.simpleFunctions
				.get(lSrc.get("activation").getAsString());

			JsonArray nestedWeight = lSrc.get("weight").getAsJsonArray(),
				nestedBias = lSrc.get("bias").getAsJsonArray();

			int weightCols = nestedWeight.get(0).getAsJsonArray().size();
			double[][] w = new double[nestedWeight.size()][weightCols], b = new double[nestedBias
				.size()][1];

			for (int i = 0; i < neurons; i++) {
				JsonArray nested = nestedWeight.get(i).getAsJsonArray();
				double[] values = new double[weightCols];
				for (int j = 0; j < weightCols; j++) {
					values[j] = nested.get(j).getAsDouble();
				}
				w[i] = values;

				b[i][0] = nestedBias.get(i).getAsDouble();
			}

			Matrix<SMatrix> weights = new SimpleMatrix(w), bias = new SimpleMatrix(b);

			NetworkLayer<SMatrix> lr = new NetworkLayer<>(neurons, l2, act, weights, bias);
			lr.setPrecedingLayer(first);
			System.out.println(lr.getNeurons());
			lr.setDeltaBias(deltaB.get(layerIndex - 1));
			lr.setDeltaWeight(deltaW.get(layerIndex - 1));

			nBuilder.layer(lr);

			first = lr;
			layerIndex++;
		}

		var evalFunction = network.get("evaluator").getAsJsonObject();
		var evalParams = Optional.of(evalFunction.get("params").getAsJsonArray());

		EvaluationFunction<SMatrix> evaluator = NetworkDataCache.simpleEvaluators
			.get(evalFunction.get("name").getAsString());
		if (evalParams.isPresent()) {
			JsonArray paramArr = evalParams.get();
			if (paramArr.size() != 0) {
				double[] paramVals = new double[paramArr.size()];
				int t = 0;
				for (var v : paramArr) {
					paramVals[t++] = v.getAsDouble();
				}
				evaluator.init(paramVals);
			}
		}

		var costFunctionObj = network.get("costfunction").getAsJsonObject();
		CostFunction<SMatrix> costFunction = NetworkDataCache.simpleCostFunctions
			.get(costFunctionObj.get("name").getAsString());

		var optimizerObj = network.get("optimizer").getAsJsonObject();
		var optimizerParams = Optional.of(optimizerObj.get("params").getAsJsonArray());

		Optimizer<SMatrix> optimizer = NetworkDataCache.simpleOptimisers
			.get(optimizerObj.get("name").getAsString());
		if (optimizerParams.isPresent()) {
			JsonArray paramArr = optimizerParams.get();
			if (paramArr.size() != 0) {
				double[] paramVals = new double[paramArr.size()];
				int t = 0;
				for (var v : paramArr) {
					paramVals[t++] = v.getAsDouble();
				}
				optimizer.init(paramVals);
			}
		}

		nBuilder.initializer(init);
		nBuilder.optimizer(optimizer);
		nBuilder.costFunction(costFunction);
		nBuilder.evaluationFunction(evaluator);
		nBuilder.clipping(network.get("clipping").getAsBoolean());

		return nBuilder.deserialize();
	};
	private final Gson gson;

	public SimpleDeserializer() {
		GsonBuilder b = new GsonBuilder();
		b.registerTypeAdapter(networkType, deserializer);
		this.gson = b.create();
	}

	public LayeredNeuralNetwork<SMatrix> deserialize(File jsonFile) {

		LayeredNeuralNetwork<SMatrix> out = null;
		try (JsonReader reader = new JsonReader(new FileReader(jsonFile))) {
			out = gson.fromJson(reader, networkType);
		} catch (IOException | JsonSyntaxException e) {
			e.printStackTrace();
		}

		return out;
	}

	public LayeredNeuralNetwork<SMatrix> deserialize(String json) {
		json = json.trim();
		return gson.fromJson(json, networkType);
	}

}
