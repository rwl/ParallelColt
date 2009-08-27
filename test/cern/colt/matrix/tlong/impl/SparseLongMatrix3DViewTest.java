package cern.colt.matrix.tlong.impl;

public class SparseLongMatrix3DViewTest extends SparseLongMatrix3DTest {

    public SparseLongMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseLongMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseLongMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}