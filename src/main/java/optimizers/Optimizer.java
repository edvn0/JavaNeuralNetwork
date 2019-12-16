package optimizers;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

public interface Optimizer extends Serializable {

	DenseMatrix[] changeWeights(DenseMatrix[] weights, DenseMatrix[] deltas);

	DenseMatrix[] changeBiases(DenseMatrix[] biases, DenseMatrix[] deltas);

	void initializeOptimizer(int layers);

}
