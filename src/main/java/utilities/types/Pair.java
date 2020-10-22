package utilities.types;

import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof Pair)) {
            return false;
        }
        Pair<L, R> pair = (Pair<L, R>) o;
        return Objects.equals(left, pair.left) && Objects.equals(right, pair.right);
    }

    @Override
    public int hashCode() {
        return Objects.hash(left, right);
    }

}
