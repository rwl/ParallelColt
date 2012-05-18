package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix1DTest;

public class SparseFloatMatrix1DTest extends FloatMatrix1DTest {

    public SparseFloatMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix1D(SIZE);
        B = new SparseFloatMatrix1D(SIZE);
    }
}
