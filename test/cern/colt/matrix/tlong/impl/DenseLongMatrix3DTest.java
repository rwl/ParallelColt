package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix3DTest;

public class DenseLongMatrix3DTest extends LongMatrix3DTest {

    public DenseLongMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLongMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLongMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

}
