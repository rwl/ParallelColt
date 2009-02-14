package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class CCFloatMatrix2DTest extends FloatMatrix2DTest {

    public CCFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCFloatMatrix2D(NROWS, NCOLUMNS);
        B = new CCFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new CCFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
