package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix1DTest;

public class DenseIntMatrix1DTest extends IntMatrix1DTest {

    public DenseIntMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix1D(SIZE);
        B = new DenseIntMatrix1D(SIZE);
    }
}