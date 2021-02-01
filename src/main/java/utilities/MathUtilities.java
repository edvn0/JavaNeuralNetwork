package utilities;

import java.util.function.Function;

public class MathUtilities {

	public static double[][] simpleMap(final Function<Double, Double> mapping,
		final double[][] doubles) {
		double[][] out = new double[doubles.length][doubles[0].length];
		for (int i = 0; i < doubles.length; i++) {
			for (int j = 0; j < doubles[0].length; j++) {
				out[i][j] = mapping.apply(doubles[i][j]);
			}
		}
		return out;
	}

	public static int argMax(final double[] array) {
		int argMax = 0;
		double best = array[0];
		for (int i = 1; i < array.length - 1; i++) {
			if (array[i] > best) {
				best = array[i];
				argMax = i;
			}
		}

		return argMax;
	}
}
