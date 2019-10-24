package math;

import java.util.HashMap;
import java.util.Map;

public class ActivationFunctionFactory {

	private Map<String, ActivationFunction> activationFunctionMap = new HashMap<>();

	public ActivationFunctionFactory() {
		// Fill map with all the activation functions
		ActivationFunction sigmoid = new SigmoidFunction();
		activationFunctionMap.put(sigmoid.getName(), sigmoid);

		ActivationFunction tanh = new TanhFunction();
		activationFunctionMap.put(tanh.getName(), tanh);

		ActivationFunction relu = new ReluFunction();
		activationFunctionMap.put(relu.getName(), relu);
	}

	public ActivationFunction getActivationFunctionByKey(String activationFunctionKey) {
		return activationFunctionMap.get(activationFunctionKey);
	}

	public void addActivationFunction(String key, ActivationFunction activationFunction) {
		activationFunctionMap.put(key, activationFunction);
	}
}