package cern.colt.matrix.tdouble.impl;


public class SparseDoubleMatrix3DViewTest extends SparseDoubleMatrix3DTest {

    public SparseDoubleMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}