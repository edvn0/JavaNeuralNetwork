package demos;

import demos.implementations.SandboxMNIST;
import demos.implementations.SandboxXOR;

public class Sandbox {

    public static void main(String[] args) {
        AbstractDemo mnist = new SandboxXOR();
        mnist.demo();
    }

}
