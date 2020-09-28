package demos;

import math.activations.TanhFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.EvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.inputs.NetworkInput;
import optimizers.StochasticGradientDescent;
import org.ujmp.core.Matrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class XORTester {

    private List<NetworkInput> data;

    private BufferedImage[] images;

    private double[][] xorData = new double[][]{{0, 1}, {0, 0}, {1, 1}, {1, 0}};
    private double[][] xorLabel = new double[][]{{1}, {0}, {0}, {1}};

    private NeuralNetwork network;
    private String path;

    private int imagesSize;
    private int w, h;

    XORTester(String path, int size) {
        this.path = path;
        this.imagesSize = size;
        w = 600;
        h = 600;
        images = new BufferedImage[imagesSize];

        data = new ArrayList<>();
        SecureRandom r = new SecureRandom();
        for (int i = 0; i < 10000; i++) {
            double[][] cData;
            double[][] cLabel;
            int rd = r.nextInt(xorData.length);
            cData = new double[][]{xorData[rd]};
            cLabel = new double[][]{xorLabel[rd]};
            data.add(new NetworkInput(Matrix.Factory.importFromArray(cData).transpose(),
                    Matrix.Factory.importFromArray(cLabel).transpose()));
        }
        Collections.shuffle(data);

        network = new NeuralNetwork(
                new NetworkBuilder(4).setFirstLayer(2).setLayer(10, new TanhFunction()).setLayer(10, new TanhFunction())
                        .setLastLayer(1, new TanhFunction()).setCostFunction(new MeanSquaredCostFunction())
                        .setEvaluationFunction((EvaluationFunction) toEvaluate -> {
                            double perc = 0;
                            for (var i : toEvaluate) {
                                double data = i.getData().doubleValue();
                                double label = i.getLabel().doubleValue();

                                System.out.println(data + " " + label);

                                perc += Math.abs(data - label) < 0.1 ? 1 : 0;
                            }
                            return perc / toEvaluate.size();
                        })
                        .setOptimizer(new StochasticGradientDescent(0.05)));

        network.display();
        network.trainWithMetrics(data.subList(0, 1000), data.subList(1000, 2000), 70, 64, "E:\\Programming\\Git\\JavaNeuralNetwork\\src\\main\\resources\\output");
    }

    public static void main(String[] args) throws IOException {
        new XORTester("C:\\Users\\edvin\\Downloads", 3);
    }

    private void run(int in) throws IOException {
        if (w % in != 0 && h % in != 0) {
            throw new IllegalArgumentException();
        }
        int cols = w / in;
        int rows = h / in;
        for (int l = 0; l < imagesSize; l++) {
            BufferedImage img = new BufferedImage(600, 600, BufferedImage.TYPE_INT_ARGB);
            images[l] = img;
            System.out.println("Image " + l);
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    double col = (double) i / cols;
                    double row = (double) j / rows;
                    double out = 0;
                    if (i % (in) == 0) {
                        out = network.predict(toInputMatrix(col, row)).doubleValue();
                    }
                    out *= 255;
                    int colors = (int) out;
                    Color c = new Color(colors, colors, colors);
                    img.setRGB(i, j, c.getRGB());
                }
            }
        }

        int k = 0;
        for (BufferedImage img : images) {
            String out = path + "/XOR_" + k++ + ".png";
            ImageIO.write(img, "png", new File(out));
        }
    }

    private Matrix toInputMatrix(final double col, final double row) {
        return Matrix.Factory.importFromArray(new double[][]{{(double) col}, {(double) row}});
    }
}