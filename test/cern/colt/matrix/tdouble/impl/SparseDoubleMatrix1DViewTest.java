package cern.colt.matrix.tdouble.impl;

public class SparseDoubleMatrix1DViewTest extends SparseDoubleMatrix1DTest {

    public SparseDoubleMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix1D(SIZE).viewFlip();
        B = new SparseDoubleMatrix1D(SIZE).viewFlip();
    }
}