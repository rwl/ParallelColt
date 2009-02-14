package cern.colt.matrix.tfloat.impl;


public class SparseFloatMatrix1DViewTest extends SparseFloatMatrix1DTest {

    public SparseFloatMatrix1DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix1D(SIZE).viewFlip();
        B = new SparseFloatMatrix1D(SIZE).viewFlip();
    }
}