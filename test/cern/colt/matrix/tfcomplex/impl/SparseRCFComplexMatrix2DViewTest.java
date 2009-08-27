package cern.colt.matrix.tfcomplex.impl;

public class SparseRCFComplexMatrix2DViewTest extends SparseRCFComplexMatrix2DTest {

    public SparseRCFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
