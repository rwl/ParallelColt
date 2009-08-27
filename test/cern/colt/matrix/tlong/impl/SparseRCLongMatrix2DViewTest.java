package cern.colt.matrix.tlong.impl;

public class SparseRCLongMatrix2DViewTest extends SparseRCLongMatrix2DTest {

    public SparseRCLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
