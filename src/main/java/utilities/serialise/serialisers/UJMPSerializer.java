package utilities.serialise.serialisers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
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
import neuralnetwork.NeuralNetwork;

public class UJMPSerializer {

    private Gson gson;
    private Type network = new TypeToken<NeuralNetwork<org.ujmp.core.Matrix>>() {
    }.getType();

    public UJMPSerializer() {

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
                new JsonSerializer<List<ActivationFunction<org.ujmp.core.Matrix>>>() {
                    @Override
                    public JsonElement serialize(List<ActivationFunction<org.ujmp.core.Matrix>> src, Type typeOfSrc,
                            JsonSerializationContext context) {
                        JsonArray arr = new JsonArray();
                        for (var m : src) {
                            arr.add(m.getName());
                        }
                        return arr;
                    }
                });

        gsonb.registerTypeAdapter(optimiser, new JsonSerializer<Optimizer<org.ujmp.core.Matrix>>() {
            @Override
            public JsonElement serialize(Optimizer<org.ujmp.core.Matrix> src, Type typeOfSrc,
                    JsonSerializationContext context) {
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
            }

        });

        gsonb.registerTypeAdapter(evalFunction, new JsonSerializer<EvaluationFunction<org.ujmp.core.Matrix>>() {
            @Override
            public JsonElement serialize(EvaluationFunction<org.ujmp.core.Matrix> src, Type typeOfSrc,
                    JsonSerializationContext context) {
                JsonObject inner = new JsonObject();
                var info = src.params();

                inner.addProperty("name", src.name());
                if (info != null)
                    info.forEach((a, b) -> inner.addProperty("value", b));

                return inner;
            }
        });

        gsonb.registerTypeAdapter(costFunction, new JsonSerializer<CostFunction<org.ujmp.core.Matrix>>() {
            @Override
            public JsonElement serialize(CostFunction<org.ujmp.core.Matrix> src, Type typeOfSrc,
                    JsonSerializationContext context) {
                JsonObject name = new JsonObject();
                name.addProperty("name", src.name());
                return name;
            }
        });

        gsonb.registerTypeAdapter(matrices, new JsonSerializer<List<Matrix<org.ujmp.core.Matrix>>>() {
            @Override
            public JsonElement serialize(List<Matrix<org.ujmp.core.Matrix>> src, Type typeOfSrc,
                    JsonSerializationContext context) {
                JsonObject matrixOut = new JsonObject();
                int layer = 0;
                for (var s : src) {
                    JsonArray arr = new JsonArray();
                    double[][] out = s.rawCopy();
                    for (double[] ds : out) {
                        JsonArray inner = new JsonArray();
                        for (double d : ds) {
                            inner.add(d);
                        }
                        arr.add(inner);
                    }

                    matrixOut.add(layer + "", arr);
                    layer++;
                }

                return matrixOut;

            }
        });

        this.gson = gsonb.create();
    }

    public void serialise(File f, NeuralNetwork<org.ujmp.core.Matrix> ojAlgoNetwork) {
        String json = serialiseToString(ojAlgoNetwork);

        try (FileWriter fw = new FileWriter(f, false)) {
            fw.write(json);
        } catch (IOException e) {

        }

    }

    public String serialiseToString(NeuralNetwork<org.ujmp.core.Matrix> ojAlgoNetwork) {
        return gson.toJson(ojAlgoNetwork, network);
    }

}
