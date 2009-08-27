package cern.colt.matrix.tdcomplex.impl;

public class SparseCCDComplexMatrix2DViewTest extends SparseCCDComplexMatrix2DTest {

    public SparseCCDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
