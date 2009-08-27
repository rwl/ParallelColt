package cern.colt.matrix.tint.impl;

public class SparseIntMatrix1DViewTest extends SparseIntMatrix1DTest {

    public SparseIntMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseIntMatrix1D(SIZE).viewFlip();
        B = new SparseIntMatrix1D(SIZE).viewFlip();
    }
}