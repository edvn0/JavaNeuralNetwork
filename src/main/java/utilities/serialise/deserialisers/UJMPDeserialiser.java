package utilities.serialise.deserialisers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
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
import math.linearalgebra.ujmp.UJMPMatrix;
import math.optimizers.Optimizer;
import neuralnetwork.NeuralNetwork;
import utilities.serialise.NetworkDataCache;
import utilities.serialise.adapters.UJMPNetworkDeserializer;

public class UJMPDeserialiser {

    private Gson gson;
    private Type network = new TypeToken<NeuralNetwork<org.ujmp.core.Matrix>>() {
    }.getType();

    public UJMPDeserialiser() {
        GsonBuilder gsonb = new GsonBuilder();

        Type activationFunctions = new TypeToken<List<ActivationFunction<org.ujmp.core.Matrix>>>() {
        }.getType();
        Type optimiser = new TypeToken<Optimizer<org.ujmp.core.Matrix>>() {
        }.getType();
        Type evalFunction = new TypeToken<EvaluationFunction<org.ujmp.core.Matrix>>() {
        }.getType();
        Type costFunction = new TypeToken<CostFunction<org.ujmp.core.Matrix>>() {
        }.getType();
        Type matrices = new TypeToken<List<Matrix<org.ujmp.core.Matrix>>>() {
        }.getType();

        gsonb.registerTypeAdapter(activationFunctions,
                new JsonDeserializer<List<ActivationFunction<org.ujmp.core.Matrix>>>() {
                    @Override
                    public List<ActivationFunction<org.ujmp.core.Matrix>> deserialize(JsonElement json, Type typeOfT,
                            JsonDeserializationContext context) throws JsonParseException {
                        JsonArray arr = json.getAsJsonArray();
                        List<ActivationFunction<org.ujmp.core.Matrix>> functions = new ArrayList<>();
                        for (var a : arr) {
                            functions.add(NetworkDataCache.ujmpFunctions.get(a.getAsString()));
                        }

                        return functions;
                    }
                });

        gsonb.registerTypeAdapter(optimiser, new JsonDeserializer<Optimizer<org.ujmp.core.Matrix>>() {

            @Override
            public Optimizer<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                Optimizer<org.ujmp.core.Matrix> op = NetworkDataCache.ujmpOptimisers.get(obj.get("name").getAsString());

                double lR = tryToFind(obj, "v1"); // always learning rate
                double v2 = tryToFind(obj, "v2"); // beta1 or momentum
                double v3 = tryToFind(obj, "v3"); // beta2

                double[] vals = new double[] { lR, v2, v3 };

                op.init(vals);

                return op;
            }
        });
        gsonb.registerTypeAdapter(costFunction, new JsonDeserializer<CostFunction<org.ujmp.core.Matrix>>() {
            @Override
            public CostFunction<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                CostFunction<org.ujmp.core.Matrix> cf = NetworkDataCache.ujmpCostFunctions
                        .get(obj.get("name").getAsString());

                return cf;
            }
        });
        gsonb.registerTypeAdapter(evalFunction, new JsonDeserializer<EvaluationFunction<org.ujmp.core.Matrix>>() {
            @Override
            public EvaluationFunction<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                EvaluationFunction<org.ujmp.core.Matrix> cf = NetworkDataCache.ujmpEvaluators
                        .get(obj.get("name").getAsString());

                double val = tryToFind(obj, "v1");

                cf.init(val);

                return cf;
            }
        });

        gsonb.registerTypeAdapter(matrices, new JsonDeserializer<List<Matrix<org.ujmp.core.Matrix>>>() {

            @Override
            public List<Matrix<org.ujmp.core.Matrix>> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject matrices = json.getAsJsonObject();
                int index = matrices.entrySet().size();
                List<Matrix<org.ujmp.core.Matrix>> out = new ArrayList<>();
                for (int i = 0; i < index; i++) {
                    JsonArray arr = matrices.get(i + "").getAsJsonArray();

                    int rows = arr.size();
                    int cols = arr.get(0).getAsJsonArray().size();

                    double[][] values = new double[rows][cols];
                    for (int k = 0; k < rows; k++) {
                        JsonArray row = arr.get(k).getAsJsonArray();
                        for (int j = 0; j < cols; j++) {
                            values[k][j] = row.get(j).getAsDouble();
                        }
                    }
                    out.add(new UJMPMatrix(values));
                }

                return out;
            }
        });

        gsonb.registerTypeAdapter(network, new UJMPNetworkDeserializer());
        this.gson = gsonb.create();
    }

    public NeuralNetwork<org.ujmp.core.Matrix> deserialise(File jsonFile) {

        NeuralNetwork<org.ujmp.core.Matrix> out = null;
        try (JsonReader reader = new JsonReader(new FileReader(jsonFile))) {
            out = gson.fromJson(reader, network);
        } catch (IOException | JsonSyntaxException e) {
            e.printStackTrace();
        }

        return out;
    }

    public NeuralNetwork<org.ujmp.core.Matrix> deserialise(String json) {
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
