package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2DTest;

public class SparseRCMLongMatrix2DTest extends LongMatrix2DTest {

    public SparseRCMLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMLongMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMLongMatrix2D(NCOLUMNS, NROWS);
    }

}
