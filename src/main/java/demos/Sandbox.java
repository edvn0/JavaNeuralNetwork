package demos;

import math.linearalgebra.simple.SMatrix;
import org.ojalgo.matrix.Primitive64Matrix;

public class Sandbox {

	public static void main(String[] args) {
		AbstractDemo<Primitive64Matrix> mnistOjAlgo = new demos.implementations.ojalgo.SandboxMnist();
		AbstractDemo<Primitive64Matrix> mnistOjAlgoLayered = new demos.implementations.ojalgo.SandboxMnistLayered();
		AbstractDemo<Primitive64Matrix> xorOjAlgo = new demos.implementations.ojalgo.SandboxXOR();
		AbstractDemo<org.ujmp.core.Matrix> mnistUJMP = new demos.implementations.ujmp.SandboxMnist();
		AbstractDemo<org.ujmp.core.Matrix> xorUJMP = new demos.implementations.ujmp.SandboxXOR();

		AbstractDemo<SMatrix> mnistSimpleLayered = new demos.implementations.simple.SandboxMnistLayered();
		AbstractDemo<SMatrix> mnistSimplePure = new demos.implementations.simple.SandboxMnistPure();

		AbstractDemo<?> demo = null;

		if (args == null || args.length == 0) {
			throw new IllegalArgumentException(
				"Supply choice of demo! args: MNIST_OJ_ALGO, XOR_OJ_ALGO, MNIST_UJMP, XOR_UJMP.");
		} else {
			String choice = args[0];

			switch (choice) {
				case "MNIST_OJ_ALGO":
					demo = mnistOjAlgo;
					break;
				case "XOR_OJ_ALGO":
					demo = xorOjAlgo;
					break;
				case "MNIST_UJMP":
					demo = mnistUJMP;
					break;
				case "XOR_UJMP":
					demo = xorUJMP;
					break;
				case "MNIST_OJ_ALGO_LAYERED":
					demo = mnistOjAlgoLayered;
					break;
				case "SIMPLE_LAYER":
					demo = mnistSimpleLayered;
					break;
				case "SIMPLE_PURE":
					demo = mnistSimplePure;
					break;
				default:
					throw new IllegalArgumentException(
						"Supply choice of demo! args: MNIST_OJ_ALGO, XOR_OJ_ALGO, MNIST_UJMP, XOR_UJMP.");
			}
		}

		demo.demo();
	}

}
