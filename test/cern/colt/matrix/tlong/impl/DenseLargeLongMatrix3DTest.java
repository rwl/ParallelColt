package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix3DTest;

public class DenseLargeLongMatrix3DTest extends LongMatrix3DTest {

    public DenseLargeLongMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeLongMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLargeLongMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}
