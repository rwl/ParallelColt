package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class SparseRCFComplexMatrix2DTest extends FComplexMatrix2DTest {

    public SparseRCFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCFComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
