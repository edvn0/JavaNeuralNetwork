package utilities.serialise.adapters;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.List;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.optimizers.Optimizer;
import neuralnetwork.layer.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.serialise.LayeredOjAlgoNetwork;

public class LayeredOjAlgoNetworkDeserializer implements
	JsonDeserializer<LayeredNeuralNetwork<Primitive64Matrix>> {

	@Override
	public LayeredNeuralNetwork<Primitive64Matrix> deserialize(JsonElement json, Type typeOfT,
		JsonDeserializationContext context) throws JsonParseException {

		Type optimiser = new TypeToken<Optimizer<Primitive64Matrix>>() {
		}.getType();
		Type evalFunction = new TypeToken<EvaluationFunction<Primitive64Matrix>>() {
		}.getType();
		Type costFunction = new TypeToken<CostFunction<Primitive64Matrix>>() {
		}.getType();
		Type layers = new TypeToken<List<NetworkLayer<Primitive64Matrix>>>() {
		}.getType();

		JsonObject nn = json.getAsJsonObject();

		CostFunction<Primitive64Matrix> cf = context
			.deserialize(nn.get("costFunction"), costFunction);
		Optimizer<Primitive64Matrix> op = context.deserialize(nn.get("optimizer"), optimiser);
		EvaluationFunction<Primitive64Matrix> ef = context
			.deserialize(nn.get("evaluationFunction"), evalFunction);

		List<NetworkLayer<Primitive64Matrix>> layerData = context
			.deserialize(nn.get("layers"), layers);

		return LayeredOjAlgoNetwork.create(layerData.get(0).getNeurons(), layerData, cf, op, ef);
	}

}
