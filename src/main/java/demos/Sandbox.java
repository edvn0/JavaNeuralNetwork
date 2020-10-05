package demos;

import demos.implementations.ujmp.SandboxXOR;

public class Sandbox {

    public static void main(String[] args) {
        AbstractDemo<org.ujmp.core.Matrix> mnist = new SandboxXOR();
        mnist.demo();
    }

}
