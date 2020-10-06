package utilities.serialise.deserialisers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
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

import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.linearalgebra.ujmp.UJMPMatrix;
import math.optimizers.Optimizer;
import neuralnetwork.NeuralNetwork;
import utilities.serialise.ConverterUtil;
import utilities.serialise.adapters.OjAlgoNetworkDeserializer;

public class UJMPDeserialiser {

    public void deserialise(String file) {
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
                            functions.add(ConverterUtil.ujmpFunctions.get(a.getAsString()));
                        }

                        return functions;
                    }
                });

        gsonb.registerTypeAdapter(optimiser, new JsonDeserializer<Optimizer<org.ujmp.core.Matrix>>() {
            @Override
            public Optimizer<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                Optimizer<org.ujmp.core.Matrix> op = ConverterUtil.ujmpOptimisers.get(obj.get("name").getAsString());
                double[] vals = new double[] { obj.get("learningRate").getAsDouble(), obj.get("beta1").getAsDouble(),
                        obj.get("beta2").getAsDouble() };

                op.init(vals);

                return op;
            }
        });
        gsonb.registerTypeAdapter(costFunction, new JsonDeserializer<CostFunction<org.ujmp.core.Matrix>>() {
            @Override
            public CostFunction<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                CostFunction<org.ujmp.core.Matrix> cf = ConverterUtil.ujmpCostFunctions
                        .get(obj.get("name").getAsString());

                return cf;
            }
        });
        gsonb.registerTypeAdapter(evalFunction, new JsonDeserializer<EvaluationFunction<org.ujmp.core.Matrix>>() {
            @Override
            public EvaluationFunction<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
                    JsonDeserializationContext context) throws JsonParseException {
                JsonObject obj = json.getAsJsonObject();

                EvaluationFunction<org.ujmp.core.Matrix> cf = ConverterUtil.ujmpEvaluators
                        .get(obj.get("name").getAsString());

                double val = obj.get("value").getAsDouble();

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

        Type network = new TypeToken<NeuralNetwork<org.ujmp.core.Matrix>>() {
        }.getType();
        gsonb.registerTypeAdapter(network, new OjAlgoNetworkDeserializer());
        gsonb.setPrettyPrinting();
        Gson gson = gsonb.create();
        try {
            JsonReader reader = new JsonReader(new FileReader(new File(file)));
            try {
                NeuralNetwork<org.ujmp.core.Matrix> out = gson.fromJson(reader, network);
                out.display();
            } catch (JsonSyntaxException exception) {
                exception.printStackTrace();
            }
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        // out.display();
    }

}
