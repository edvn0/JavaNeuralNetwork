package demos;

import org.ojalgo.matrix.Primitive64Matrix;

import demos.implementations.ojalgo.SandboxMnist;

public class Sandbox {

    public static void main(String[] args) {
        AbstractDemo<Primitive64Matrix> mnist = new SandboxMnist();
        mnist.demo();
    }

}
