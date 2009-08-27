package cern.colt.matrix.tfcomplex.impl;

public class SparseRCMFComplexMatrix2DViewTest extends SparseRCMFComplexMatrix2DTest {

    public SparseRCMFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        B = new SparseRCMFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        Bt = new SparseRCMFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
    }

}
