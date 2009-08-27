package cern.colt.matrix.tint.impl;

public class SparseIntMatrix3DViewTest extends SparseIntMatrix3DTest {

    public SparseIntMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}