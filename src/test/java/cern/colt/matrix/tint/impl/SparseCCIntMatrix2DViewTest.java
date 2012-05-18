package cern.colt.matrix.tint.impl;

public class SparseCCIntMatrix2DViewTest extends SparseCCIntMatrix2DTest {

    public SparseCCIntMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCIntMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCIntMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
