package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix3DTest;

public class SparseDoubleMatrix3DTest extends DoubleMatrix3DTest {

    public SparseDoubleMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new SparseDoubleMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }
}