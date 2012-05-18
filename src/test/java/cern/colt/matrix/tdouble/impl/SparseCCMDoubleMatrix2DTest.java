package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class SparseCCMDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public SparseCCMDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCMDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCMDoubleMatrix2D(NCOLUMNS, NROWS);
    }

}
