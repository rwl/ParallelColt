package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix3DTest;

public class SparseFComplexMatrix3DTest extends FComplexMatrix3DTest {
    public SparseFComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new SparseFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}
