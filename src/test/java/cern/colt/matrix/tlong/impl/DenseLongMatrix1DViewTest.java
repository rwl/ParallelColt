package cern.colt.matrix.tlong.impl;

public class DenseLongMatrix1DViewTest extends DenseLongMatrix1DTest {

    public DenseLongMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLongMatrix1D(SIZE).viewFlip();
        B = new DenseLongMatrix1D(SIZE).viewFlip();
    }
}