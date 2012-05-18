package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class SparseCCFComplexMatrix2DTest extends FComplexMatrix2DTest {

    public SparseCCFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCFComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
