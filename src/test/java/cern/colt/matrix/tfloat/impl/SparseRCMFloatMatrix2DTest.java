package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class SparseRCMFloatMatrix2DTest extends FloatMatrix2DTest {

    public SparseRCMFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMFloatMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCMFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCMFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
