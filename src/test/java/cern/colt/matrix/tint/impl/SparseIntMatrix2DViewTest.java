package cern.colt.matrix.tint.impl;

public class SparseIntMatrix2DViewTest extends SparseIntMatrix2DTest {

    public SparseIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}