package cern.colt.matrix.tdcomplex.impl;

public class SparseRCDComplexMatrix2DViewTest extends SparseRCDComplexMatrix2DTest {

    public SparseRCDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
