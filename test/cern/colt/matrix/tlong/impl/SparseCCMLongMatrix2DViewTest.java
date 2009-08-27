package cern.colt.matrix.tlong.impl;

public class SparseCCMLongMatrix2DViewTest extends SparseCCMLongMatrix2DTest {

    public SparseCCMLongMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCMLongMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCMLongMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
