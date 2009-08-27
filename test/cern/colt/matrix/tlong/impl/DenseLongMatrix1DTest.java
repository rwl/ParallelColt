package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix1DTest;

public class DenseLongMatrix1DTest extends LongMatrix1DTest {

    public DenseLongMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLongMatrix1D(SIZE);
        B = new DenseLongMatrix1D(SIZE);
    }
}