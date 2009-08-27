package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2DTest;

public class SparseRCMIntMatrix2DTest extends IntMatrix2DTest {

    public SparseRCMIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMIntMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMIntMatrix2D(NCOLUMNS, NROWS);
    }

}
