package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class SparseFComplexMatrix2DTest extends FComplexMatrix2DTest {
    public SparseFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new SparseFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseFComplexMatrix2D(NCOLUMNS, NROWS);
    }

}
