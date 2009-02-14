package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix1DTest;

public class SparseDoubleMatrix1DTest extends DoubleMatrix1DTest {

    public SparseDoubleMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix1D(SIZE);
        B = new SparseDoubleMatrix1D(SIZE);
    }
}