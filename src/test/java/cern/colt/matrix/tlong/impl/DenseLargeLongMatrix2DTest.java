package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2DTest;

public class DenseLargeLongMatrix2DTest extends LongMatrix2DTest {

    public DenseLargeLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeLongMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLargeLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLargeLongMatrix2D(NCOLUMNS, NROWS);
    }
}
