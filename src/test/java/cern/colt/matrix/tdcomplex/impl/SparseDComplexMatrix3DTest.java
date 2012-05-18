package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix3DTest;

public class SparseDComplexMatrix3DTest extends DComplexMatrix3DTest {
    public SparseDComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new SparseDComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}
