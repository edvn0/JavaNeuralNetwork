package utilities.serialise.serialisers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Objects;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;

import math.linearalgebra.simple.SMatrix;
import neuralnetwork.LayeredNeuralNetwork;

public class SimpleSerializer {

    private Gson gson;
    private Type networkType = new TypeToken<LayeredNeuralNetwork<SMatrix>>() {
    }.getType();

    public SimpleSerializer() {
        GsonBuilder builder = new GsonBuilder();
        builder.registerTypeAdapter(networkType, adapter);
        gson = builder.create();
    }

    public void serialize(final File fileName, LayeredNeuralNetwork<SMatrix> network) {
        String json = gson.toJson(network, this.networkType);

        try (FileWriter fw = new FileWriter(fileName, false)) {
            fw.write(json);
        } catch (IOException e) {

        }
    }

    private static JsonSerializer<LayeredNeuralNetwork<SMatrix>> adapter = (src, type, context) -> {
        JsonObject networkSerialisation = new JsonObject();
        // layers
        var layers = src.getLayers();

        JsonArray layersArray = new JsonArray();
        JsonObject firstL = new JsonObject();
        var firstLayer = layers.remove(0);

        firstL.addProperty("neurons", firstLayer.getNeurons());

        for (var l : layers) {
            JsonObject layer = new JsonObject();
            System.out.println("Layer: " + l);

            // activation
            // weight
            // bias
            // neurons
            layer.addProperty("neurons", l.getNeurons());
            layer.addProperty("activation", l.getFunction().getName());

            JsonArray weights = new JsonArray(), biases = new JsonArray();
            double[][] w = l.getWeight().rawCopy(), b = l.getBias().rawCopy();

            for (double[] ds : w) {
                JsonArray inner = new JsonArray();
                for (double d : ds) {
                    inner.add(new JsonPrimitive(d));
                }
                weights.add(inner);
            }
            for (double[] ds : b) {
                JsonArray inner = new JsonArray();
                for (double d : ds) {
                    inner.add(new JsonPrimitive(d));
                }
                biases.add(inner);
            }

            layer.add("weight", weights);
            layer.add("bias", biases);
            layersArray.add(layer);
        }

        networkSerialisation.add("layers", layersArray);
        // end layers

        // start optimizer
        JsonObject optimizer = new JsonObject();
        optimizer.addProperty("name", src.getOptimizer().name());
        JsonArray parameters = new JsonArray();
        if (src.getOptimizer().params() != null) {
            src.getOptimizer().params().values().stream().filter(Objects::nonNull)
                    .forEach((e) -> parameters.add(new JsonPrimitive(e)));
        }
        optimizer.add("params", parameters);
        networkSerialisation.add("optimizer", optimizer);
        // end optimizer

        // start initializer
        var init = src.getInitializer();
        JsonObject initializer = new JsonObject();
        initializer.addProperty("name", init.name());
        initializer.addProperty("weightmethod", init.getMethods().left().getName());
        initializer.addProperty("biasmethod", init.getMethods().right().getName());
        networkSerialisation.add("initializer", initializer);
        // end initializer

        // start evaluator
        var evaluator = src.getEvaluationFunction();
        JsonObject evaluatorObj = new JsonObject();
        JsonArray params = new JsonArray();
        if (evaluator.params() != null) {
            evaluator.params().values().stream().filter(Objects::nonNull).peek(System.out::println)
                    .forEach(e -> params.add(e));
        }
        evaluatorObj.addProperty("name", evaluator.name());
        evaluatorObj.add("params", params);
        networkSerialisation.add("evaluator", evaluatorObj);
        // end evaluator

        // start cost function
        var cost = src.getCostFunction();
        JsonObject costfunction = new JsonObject();
        costfunction.addProperty("name", cost.name());
        networkSerialisation.add("costfunction", costfunction);
        // end cost function

        return networkSerialisation;
    };

}
