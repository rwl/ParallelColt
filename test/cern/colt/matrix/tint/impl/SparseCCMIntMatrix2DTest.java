package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2DTest;

public class SparseCCMIntMatrix2DTest extends IntMatrix2DTest {

    public SparseCCMIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMIntMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCMIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCMIntMatrix2D(NCOLUMNS, NROWS);
    }

}
