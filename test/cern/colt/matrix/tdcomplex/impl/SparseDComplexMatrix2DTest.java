package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;

public class SparseDComplexMatrix2DTest extends DComplexMatrix2DTest {
    public SparseDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseDComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseDComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
