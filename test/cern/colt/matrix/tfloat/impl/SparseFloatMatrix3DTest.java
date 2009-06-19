package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix3DTest;

public class SparseFloatMatrix3DTest extends FloatMatrix3DTest {

    public SparseFloatMatrix3DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new SparseFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}
