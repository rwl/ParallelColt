package cern.colt.matrix.tint.impl;

public class DenseIntMatrix1DViewTest extends DenseIntMatrix1DTest {

    public DenseIntMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix1D(SIZE).viewFlip();
        B = new DenseIntMatrix1D(SIZE).viewFlip();
    }
}