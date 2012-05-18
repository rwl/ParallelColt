package cern.colt.matrix.tint.impl;

public class DenseIntMatrix3DViewTest extends DenseIntMatrix3DTest {

    public DenseIntMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
