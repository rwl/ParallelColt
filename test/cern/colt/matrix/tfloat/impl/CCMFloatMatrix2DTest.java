package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class CCMFloatMatrix2DTest extends FloatMatrix2DTest {

    public CCMFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCMFloatMatrix2D(NROWS, NCOLUMNS);
        B = new CCMFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new CCMFloatMatrix2D(NCOLUMNS, NROWS);
    }

}
