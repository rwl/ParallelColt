package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class SparseCCMFloatMatrix2DTest extends FloatMatrix2DTest {

    public SparseCCMFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMFloatMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCMFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCMFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
