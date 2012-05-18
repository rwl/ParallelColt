package cern.colt.matrix.tlong.impl;

public class SparseRCMLongMatrix2DViewTest extends SparseRCMLongMatrix2DTest {

    public SparseRCMLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCMLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCMLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCMLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
