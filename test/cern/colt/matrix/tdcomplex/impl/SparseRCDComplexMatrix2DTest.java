package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;

public class SparseRCDComplexMatrix2DTest extends DComplexMatrix2DTest {

    public SparseRCDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCDComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCDComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCDComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
