import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import math.ActivationFunction;
import math.LinearFunction;
import math.MeanSquaredErrorFunction;
import math.TanhFunction;
import matrix.Matrix;
import neuralnetwork.NeuralNetwork;

public class WHTester {

	public static double maxW = 0;
	public static double maxH = 0;

	public static void main(String[] args) throws IOException {
		ActivationFunction[] functions = new ActivationFunction[2 + 6];
		functions[0] = new LinearFunction();
		functions[1] = new TanhFunction();
		functions[2] = new TanhFunction();
		functions[3] = new TanhFunction();
		functions[4] = new TanhFunction();
		functions[5] = new TanhFunction();
		functions[6] = new TanhFunction();
		functions[7] = new TanhFunction();
		NeuralNetwork network = new NeuralNetwork(0.15, functions,
			new MeanSquaredErrorFunction(), new int[]{2, 8, 7, 6, 7, 1});

		Person[] persons = getPeople();

		List<Person> personList = Arrays.asList(persons);
		Collections.shuffle(personList);

		int train = (int) (0.6d * personList.size());

		List<Matrix[]> trainingData = toMatrixForm(personList.subList(0, train));
		List<Matrix[]> testData = toMatrixForm(personList.subList(train, personList.size() - 1));

		network.displayWeights();
		network.stochasticGradientDescent(trainingData, testData, 200, 32);
	}

	private static List<Matrix[]> toMatrixForm(List<Person> subList) {
		List<Matrix[]> matrices = new ArrayList<>();

		for (Person p : subList) {
			matrices.add(getDataFromPerson(p));
		}

		return matrices;
	}

	private static Person[] getPeople() throws IOException {
		List<String> data = Files.readAllLines(Paths.get(
			"/Users/edwincarlsson/Documents/Programmering/GradleProjects/NeuralNetwork/src/test/resources/weight-height.csv"));
		Person[] persons = new Person[data.size() - 1];
		int k = 0;
		for (String s : data) {
			if (!s.equals("\"Gender\",\"Height\",\"Weight\"")) {
				String[] info = s.split(",");
				persons[k++] = new Person(info[0], Double.parseDouble(info[1]),
					Double.parseDouble(info[1]));
			}
		}
		maxW = getMax(persons, true);
		maxH = getMax(persons, false);
		return persons;
	}

	private static double getMax(Person[] persons, boolean isW) {
		double m = 0;
		for (Person p : persons) {
			if (isW) {
				if (p.getWeight() > m) {
					m = p.getWeight();
				}
			} else {
				if (p.getHeight() > m) {
					m = p.getHeight();
				}
			}
		}
		return m;
	}

	private static Matrix[] getDataFromPerson(Person p) {
		Matrix[] data = new Matrix[2];

		double[][] input = new double[2][1];
		input[0] = new double[]{p.getWeight() / maxW};
		input[1] = new double[]{p.getHeight() / maxH};

		data[0] = new Matrix(input); // data
		data[1] = new Matrix(new double[][]{{p.gender}}); // label
		return data;
	}

	private static class Person {

		private int gender;
		private double height, weight;

		Person(String gender, double height, double weight) {
			this.height = height * 2.54;
			this.weight = weight * 0.45359237;
			this.gender = ("\"Male\"".equals(gender)) ? 1 : 0;
		}

		public int getGender() {
			return gender;
		}

		public double getHeight() {
			return height;
		}

		public double getWeight() {
			return weight;
		}
	}

}
