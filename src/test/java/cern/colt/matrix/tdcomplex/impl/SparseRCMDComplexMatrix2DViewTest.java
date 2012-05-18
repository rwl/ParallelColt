package cern.colt.matrix.tdcomplex.impl;

public class SparseRCMDComplexMatrix2DViewTest extends SparseRCMDComplexMatrix2DTest {

    public SparseRCMDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        B = new SparseRCMDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        Bt = new SparseRCMDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
    }

}
