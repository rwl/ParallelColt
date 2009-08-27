package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix3DTest;

public class DenseLargeIntMatrix3DTest extends IntMatrix3DTest {

    public DenseLargeIntMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLargeIntMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}
