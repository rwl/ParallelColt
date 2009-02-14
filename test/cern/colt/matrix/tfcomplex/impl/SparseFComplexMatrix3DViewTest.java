package cern.colt.matrix.tfcomplex.impl;

public class SparseFComplexMatrix3DViewTest extends SparseFComplexMatrix3DTest {
    public SparseFComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new SparseFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
