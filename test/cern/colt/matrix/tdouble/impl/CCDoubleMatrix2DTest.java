package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class CCDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public CCDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new CCDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new CCDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
