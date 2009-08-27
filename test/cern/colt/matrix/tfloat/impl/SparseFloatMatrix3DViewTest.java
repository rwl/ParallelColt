package cern.colt.matrix.tfloat.impl;

public class SparseFloatMatrix3DViewTest extends SparseFloatMatrix3DTest {

    public SparseFloatMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}
