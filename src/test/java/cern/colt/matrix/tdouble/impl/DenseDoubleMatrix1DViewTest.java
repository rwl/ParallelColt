package cern.colt.matrix.tdouble.impl;

public class DenseDoubleMatrix1DViewTest extends DenseDoubleMatrix1DTest {

    public DenseDoubleMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseDoubleMatrix1D(SIZE).viewFlip();
        B = new DenseDoubleMatrix1D(SIZE).viewFlip();
    }
}