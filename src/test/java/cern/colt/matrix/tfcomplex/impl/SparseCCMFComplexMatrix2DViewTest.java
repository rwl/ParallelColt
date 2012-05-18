package cern.colt.matrix.tfcomplex.impl;

public class SparseCCMFComplexMatrix2DViewTest extends SparseCCMFComplexMatrix2DTest {

    public SparseCCMFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        B = new SparseCCMFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        Bt = new SparseCCMFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
    }

}
