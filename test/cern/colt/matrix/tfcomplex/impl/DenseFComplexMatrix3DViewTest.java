package cern.colt.matrix.tfcomplex.impl;

public class DenseFComplexMatrix3DViewTest extends DenseFComplexMatrix3DTest {
    public DenseFComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
