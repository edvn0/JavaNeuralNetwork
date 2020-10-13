package utilities.serialise.adapters;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.List;
import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.NeuralNetwork;
import utilities.serialise.UJMPNetwork;

public class UJMPNetworkDeserializer implements JsonDeserializer<NeuralNetwork<org.ujmp.core.Matrix>> {

    @Override
    public NeuralNetwork<org.ujmp.core.Matrix> deserialize(JsonElement json, Type typeOfT,
            JsonDeserializationContext context) throws JsonParseException {

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

        JsonObject nn = json.getAsJsonObject();

        int totalLayers = context.deserialize(nn.get("totalLayers"), int.class);
        int[] sizes = context.deserialize(nn.get("sizes"), int[].class);

        List<ActivationFunction<org.ujmp.core.Matrix>> functions = context.deserialize(nn.get("functions"),
                activationFunctions);
        CostFunction<org.ujmp.core.Matrix> cf = context.deserialize(nn.get("costFunction"), costFunction);
        Optimizer<org.ujmp.core.Matrix> op = context.deserialize(nn.get("optimizer"), optimiser);
        EvaluationFunction<org.ujmp.core.Matrix> ef = context.deserialize(nn.get("evaluationFunction"), evalFunction);

        List<Matrix<org.ujmp.core.Matrix>> weights = context.deserialize(nn.get("weights"), matrices);
        List<Matrix<org.ujmp.core.Matrix>> biases = context.deserialize(nn.get("biases"), matrices);

        return UJMPNetwork.create(weights, biases, totalLayers + 1, sizes, functions, cf, op, ef);
    }

}
