package cern.colt.matrix.tint.impl;

public class SparseRCIntMatrix2DViewTest extends SparseRCIntMatrix2DTest {

    public SparseRCIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
