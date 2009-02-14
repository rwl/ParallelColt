package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class RCMFloatMatrix2DTest extends FloatMatrix2DTest {

    public RCMFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCMFloatMatrix2D(NROWS, NCOLUMNS);
        B = new RCMFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new RCMFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
