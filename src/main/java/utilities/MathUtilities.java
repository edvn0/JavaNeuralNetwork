package utilities;

import java.util.function.Function;

public class MathUtilities {

	public static double[][] simpleMap(final Function<Double, Double> mapping, final double[][] doubles) {
		double[][] elements = doubles;
		double[][] out = new double[elements.length][elements[0].length];
		for (int i = 0; i < elements.length; i++) {
			for (int j = 0; j < elements[0].length; j++) {
				out[i][j] = mapping.apply(elements[i][j]);
			}
		}
		return out;
	}

	public static int argMax(final double[] array) {
		int argMax = -1;
		double best = -Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > best) {
				best = array[i];
				argMax = i;
			}
		}

		return argMax;
	}
}
