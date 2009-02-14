package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class RCDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public RCDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new RCDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new RCDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
