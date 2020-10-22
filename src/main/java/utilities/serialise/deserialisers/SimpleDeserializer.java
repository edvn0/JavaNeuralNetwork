package utilities.serialise.deserialisers;

import java.io.File;
import java.lang.reflect.Type;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializer;
import com.google.gson.reflect.TypeToken;

import math.activations.DoNothingFunction;
import math.linearalgebra.simple.SMatrix;
import neuralnetwork.LayeredNetworkBuilder;
import neuralnetwork.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;

public class SimpleDeserializer {

    private static Type networkType = new TypeToken<LayeredNeuralNetwork<SMatrix>>() {
    }.getType();
    private Gson gson;

    public SimpleDeserializer() {
        GsonBuilder b = new GsonBuilder();
        b.registerTypeAdapter(networkType, deserializer);
        this.gson = b.create();
    }

    private static final JsonDeserializer<LayeredNeuralNetwork<SMatrix>> deserializer = (src, type, context) -> {
        // "layers" ("neurons") or ("neurons", "activation", "weight", "bias")
        // "optimizer" ("name") or ("name", "params")
        // "initializer" ("name", "weightmethod", "biasmethod")
        // "evaluator" ("name") or ("name", "params")
        // "costfunction" ("name")

        var network = src.getAsJsonObject();
        var layers = network.get("layers").getAsJsonArray();
        var firstLayer = layers.remove(0);
        LayeredNetworkBuilder<SMatrix> nBuilder = new LayeredNetworkBuilder<SMatrix>(
                firstLayer.getAsJsonObject().get("neurons").getAsInt());
        NetworkLayer<SMatrix> first = new NetworkLayer<>(new DoNothingFunction<SMatrix>(),
                firstLayer.getAsJsonObject().get("neurons").getAsInt());

        return null;
    };

    public LayeredNeuralNetwork<SMatrix> deserialise(File file) {
        return null;
    }

}
