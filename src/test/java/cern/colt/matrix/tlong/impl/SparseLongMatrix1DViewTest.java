package cern.colt.matrix.tlong.impl;

public class SparseLongMatrix1DViewTest extends SparseLongMatrix1DTest {

    public SparseLongMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseLongMatrix1D(SIZE).viewFlip();
        B = new SparseLongMatrix1D(SIZE).viewFlip();
    }
}