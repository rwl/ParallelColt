package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class SparseCCMFComplexMatrix2DTest extends FComplexMatrix2DTest {

    public SparseCCMFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCMFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCMFComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
