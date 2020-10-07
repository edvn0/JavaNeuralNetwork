package utilities.serialise.deserialisers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;

import java.util.ArrayList;
import java.util.List;

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

import org.ojalgo.matrix.Primitive64Matrix;

import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.optimizers.Optimizer;
import neuralnetwork.NeuralNetwork;
import utilities.serialise.ConverterUtil;
import utilities.serialise.adapters.OjAlgoNetworkDeserializer;

public class OjAlgoDeserialiser {

    private Gson gson;
    private Type network = new TypeToken<NeuralNetwork<Primitive64Matrix>>() {
    }.getType();

    public OjAlgoDeserialiser() {
        GsonBuilder gsonb = new GsonBuilder();

        Type activationFunctions = new TypeToken<List<ActivationFunction<Primitive64Matrix>>>() {
        }.getType();
        Type optimiser = new TypeToken<Optimizer<Primitive64Matrix>>() {
        }.getType();
        Type evalFunction = new TypeToken<EvaluationFunction<Primitive64Matrix>>() {
        }.getType();
        Type costFunction = new TypeToken<CostFunction<Primitive64Matrix>>() {
        }.getType();
        Type matrices = new TypeToken<List<Matrix<Primitive64Matrix>>>() {
        }.getType();

        gsonb.registerTypeAdapter(activationFunctions,
                new JsonDeserializer<List<ActivationFunction<Primitive64Matrix>>>() {
                    @Override
                    public List<ActivationFunction<Primitive64Matrix>> deserialize(JsonElement json, Type typeOfT,
                            JsonDeserializationContext context) throws JsonParseException {
                        JsonArray arr = json.getAsJsonArray();
                        List<ActivationFunction<Primitive64Matrix>> functions = new ArrayList<>();
                        for (var a : arr) {
                            functions.add(ConverterUtil.ojFunctions.get(a.getAsString()));
                        }

                        return functions;
                    }
                });

        gsonb.registerTypeAdapter(optimiser, new JsonDeserializer<Optimizer<Primitive64Matrix>>() {

            @Override
            public Optimizer<Primitive64Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                Optimizer<Primitive64Matrix> op = ConverterUtil.ojOptimisers.get(obj.get("name").getAsString());

                double lR = tryToFind(obj, "v1"); // always learning rate
                double v2 = tryToFind(obj, "v2"); // beta1 or momentum
                double v3 = tryToFind(obj, "v3"); // beta2

                double[] vals = new double[] { lR, v2, v3 };

                op.init(vals);

                return op;
            }
        });
        gsonb.registerTypeAdapter(costFunction, new JsonDeserializer<CostFunction<Primitive64Matrix>>() {
            @Override
            public CostFunction<Primitive64Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                CostFunction<Primitive64Matrix> cf = ConverterUtil.ojCostFunctions.get(obj.get("name").getAsString());

                return cf;
            }
        });
        gsonb.registerTypeAdapter(evalFunction, new JsonDeserializer<EvaluationFunction<Primitive64Matrix>>() {
            @Override
            public EvaluationFunction<Primitive64Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                EvaluationFunction<Primitive64Matrix> cf = ConverterUtil.ojEvaluators
                        .get(obj.get("name").getAsString());

                double val = tryToFind(obj, "v1");

                cf.init(val);

                return cf;
            }
        });

        gsonb.registerTypeAdapter(matrices, new JsonDeserializer<List<Matrix<Primitive64Matrix>>>() {

            @Override
            public List<Matrix<Primitive64Matrix>> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject matrices = json.getAsJsonObject();
                int index = matrices.entrySet().size();
                List<Matrix<Primitive64Matrix>> out = new ArrayList<>();
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
                    out.add(new OjAlgoMatrix(values));
                }

                return out;
            }
        });

        gsonb.registerTypeAdapter(network, new OjAlgoNetworkDeserializer());
        this.gson = gsonb.create();
    }

    public NeuralNetwork<Primitive64Matrix> deserialise(File jsonFile) {

        NeuralNetwork<Primitive64Matrix> out = null;
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
