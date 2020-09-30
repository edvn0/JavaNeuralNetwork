package utilities.types;

public class Triple<L, M, R> {

    private final L left;
    private final M middle;
    private final R right;

    private Triple(L left, M middle, R right) {
        this.left = left;
        this.middle = middle;
        this.right = right;
    }

    public R getRight() {
        return right;
    }

    public M getMiddle() {
        return middle;
    }

    public L getLeft() {
        return left;
    }

    public static <L, M, R> Triple<L, M, R> of(L left, M middle, R right) {
        return new Triple<>(left, middle, right);
    }

}
