package utilities.types;

public class Pair<L, R> {

    private final L left;
    private final R right;

    private Pair(L left, R right) {
        this.left = left;
        this.right = right;
    }

    public static <L, R> Pair<L, R> of(L left, R right) {
        if (left == null || right == null)
            throw new IllegalArgumentException("You need to supply non-null values.");

        return new Pair<>(left, right);
    }

    public R right() {
        return right;
    }

    public L left() {
        return left;
    }

}
