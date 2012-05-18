package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class SparseRCMFComplexMatrix2DTest extends FComplexMatrix2DTest {

    public SparseRCMFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMFComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
