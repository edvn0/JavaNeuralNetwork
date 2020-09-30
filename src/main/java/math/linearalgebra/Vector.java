package math.linearalgebra;

public abstract class Vector<M> extends Matrix<M> {

    @Override
    public int cols() {
        return 1;
    }

}
