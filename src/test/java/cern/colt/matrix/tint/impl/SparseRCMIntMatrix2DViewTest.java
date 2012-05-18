package cern.colt.matrix.tint.impl;

public class SparseRCMIntMatrix2DViewTest extends SparseRCMIntMatrix2DTest {

    public SparseRCMIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCMIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCMIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
