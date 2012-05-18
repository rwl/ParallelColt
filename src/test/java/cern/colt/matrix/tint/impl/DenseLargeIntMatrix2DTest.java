package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2DTest;

public class DenseLargeIntMatrix2DTest extends IntMatrix2DTest {

    public DenseLargeIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeIntMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLargeIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLargeIntMatrix2D(NCOLUMNS, NROWS);
    }
}
