package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix3DTest;

public class SparseIntMatrix3DTest extends IntMatrix3DTest {

    public SparseIntMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new SparseIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}