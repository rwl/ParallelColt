package cern.colt.matrix.tlong.impl;

public class SparseLongMatrix2DViewTest extends SparseLongMatrix2DTest {

    public SparseLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}