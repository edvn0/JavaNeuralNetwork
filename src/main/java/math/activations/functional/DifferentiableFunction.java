package math.activations.functional;


public interface DifferentiableFunction<M> {

    M function(M m);

    M derivative(M m);

}
