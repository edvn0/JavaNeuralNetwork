package math.optimizer;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.costfunctions.MeanSquaredCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;
import math.optimizers.ADAM;
import math.optimizers.Momentum;
import math.optimizers.StochasticGradientDescent;
import neuralnetwork.layer.LayeredNetworkBuilder;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.SimpleInitializer;
import neuralnetwork.inputs.NetworkInput;
import neuralnetwork.layer.NetworkLayer;
import org.apache.log4j.BasicConfigurator;
import org.junit.Ignore;
import org.junit.Test;

@Slf4j
@Ignore
public class MomentumTest {

	{
		BasicConfigurator.configure();
	}

	private List<NetworkInput<SMatrix>> getData() {

		double[][] xData = {{0, 1}, {1, 0}, {1, 1}, {0, 0}};
		double[][] yData = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};

		Random r = new SecureRandom();

		List<NetworkInput<SMatrix>> data = new ArrayList<>();
		for (int i = 0; i < 1000; i++) {

			int rI = r.nextInt(4);

			SimpleMatrix d = new SimpleMatrix(xData[rI]);
			SimpleMatrix l = new SimpleMatrix(yData[rI]);

			NetworkInput<SMatrix> ni = new NetworkInput<>(d, l);
			data.add(ni);
		}

		return data;
	}

	@Test
	public void testMomentum() {

		var optimizer = new Momentum<SMatrix>(0.0001, 0.9);
		var cost = new MeanSquaredCostFunction<SMatrix>();
		var evaluator = new ArgMaxEvaluationFunction<SMatrix>();
		var init = new SimpleInitializer(MethodConstants.XAVIER, MethodConstants.SCALAR);
		var act = new LeakyReluFunction<SMatrix>(0.01);

		var builder = new LayeredNetworkBuilder<SMatrix>(2).optimizer(optimizer).costFunction(cost)
			.evaluationFunction(evaluator).initializer(init).layer(new NetworkLayer<>(act, 2))
			.layer(new NetworkLayer<>(act, 10, 0.01))
			.layer(new NetworkLayer<>(new SoftmaxFunction<>(), 2));

		var nnMomentum = builder.create();

		builder.optimizer(new ADAM<SMatrix>(0.001, 0.9, 0.999));

		var nnAdam = builder.create();

		builder.optimizer(new StochasticGradientDescent<SMatrix>(0.0001));

		var nnSGD = builder.create();

		var d = getData();

		double lossMomentum = nnMomentum.testLoss(d);
		double lossAdam = nnAdam.testLoss(d);
		double lossSGD = nnSGD.testLoss(d);

		log.info("Prior to training.");
		log.info("\nMomentum Loss: {}, \n Adam Loss: {}, \n SGD Loss: {}", lossMomentum, lossAdam,
			lossSGD);

		new Thread(() -> {
			double lsgd = 0;
			for (int i = 0; i < 35; i++) {
				nnAdam.train(d, d, 3, 4, true);
				lsgd += nnAdam.testLoss(d);
			}
			System.out.println("Adam: " + lsgd / 35);
		}).run();

		new Thread(() -> {
			double lsgd = 0;
			for (int i = 0; i < 35; i++) {
				nnMomentum.train(d, d, 3, 4, true);
				lsgd += nnMomentum.testLoss(d);
			}
			System.out.println("Momentum: " + lsgd / 35);
		}).run();

		new Thread(() -> {
			double lsgd = 0;
			for (int i = 0; i < 35; i++) {
				nnSGD.train(d, d, 3, 4, true);
				lsgd += nnSGD.testLoss(d);
			}
			System.out.println("SGD: " + lsgd / 35);
		}).run();

		lossMomentum = nnMomentum.testLoss(d);
		lossAdam = nnAdam.testLoss(d);
		lossSGD = nnSGD.testLoss(d);

		log.info("After training.");
		log.info("\nMomentum Loss: {}, \n Adam Loss: {}, \n SGD Loss: {}", lossMomentum, lossAdam,
			lossSGD);
	}

}
