package utilities.serialise.adapters;

import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;

import org.ojalgo.matrix.Primitive64Matrix;

import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import utilities.serialise.OjAlgoNetwork;

public class OjAlgoNetworkDeserializer implements JsonDeserializer<NeuralNetwork<Primitive64Matrix>> {

    @Override
    public NeuralNetwork<Primitive64Matrix> deserialize(JsonElement json, Type typeOfT,
            JsonDeserializationContext context) throws JsonParseException {

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

        JsonObject nn = json.getAsJsonObject();

        int totalLayers = context.deserialize(nn.get("totalLayers"), int.class);
        int[] sizes = context.deserialize(nn.get("sizes"), int[].class);

        List<ActivationFunction<Primitive64Matrix>> functions = context.deserialize(nn.get("functions"),
                activationFunctions);
        CostFunction<Primitive64Matrix> cf = context.deserialize(nn.get("costFunction"), costFunction);
        Optimizer<Primitive64Matrix> op = context.deserialize(nn.get("optimizer"), optimiser);
        EvaluationFunction<Primitive64Matrix> ef = context.deserialize(nn.get("evaluationFunction"), evalFunction);

        List<Matrix<Primitive64Matrix>> weights = context.deserialize(nn.get("weights"), matrices);
        List<Matrix<Primitive64Matrix>> biases = context.deserialize(nn.get("biases"), matrices);

        return OjAlgoNetwork.create(weights, biases, totalLayers + 1, sizes, functions, cf, op, ef);
    }

}
