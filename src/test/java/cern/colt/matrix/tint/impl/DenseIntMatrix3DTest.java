package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix3DTest;

public class DenseIntMatrix3DTest extends IntMatrix3DTest {

    public DenseIntMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

}
