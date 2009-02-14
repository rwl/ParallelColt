package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class CCMDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public CCMDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCMDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new CCMDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new CCMDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
