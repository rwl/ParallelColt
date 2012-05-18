package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class SparseRCMDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public SparseRCMDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
