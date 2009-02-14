package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class RCFloatMatrix2DTest extends FloatMatrix2DTest {

    public RCFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCFloatMatrix2D(NROWS, NCOLUMNS);
        B = new RCFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new RCFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
