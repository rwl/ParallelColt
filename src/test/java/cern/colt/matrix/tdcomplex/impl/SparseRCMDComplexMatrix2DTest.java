package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;

public class SparseRCMDComplexMatrix2DTest extends DComplexMatrix2DTest {

    public SparseRCMDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMDComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMDComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMDComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
