package cern.colt.matrix.tdcomplex.impl;

public class SparseDComplexMatrix3DViewTest extends SparseDComplexMatrix3DTest {
    public SparseDComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
