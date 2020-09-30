package demos;

import demos.implementations.SandboxXOR;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class Sandbox {

    public static void main(String[] args) {
        AbstractDemo<OjAlgoMatrix> mnist = new SandboxXOR();
        mnist.demo();
    }

}
