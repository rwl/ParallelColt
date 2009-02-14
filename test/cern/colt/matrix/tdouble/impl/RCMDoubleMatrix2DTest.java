package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class RCMDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public RCMDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCMDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new RCMDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new RCMDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
