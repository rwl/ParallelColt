package cern.colt.matrix.tdcomplex.impl;

public class SparseCCMDComplexMatrix2DViewTest extends SparseCCMDComplexMatrix2DTest {

    public SparseCCMDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCMDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        B = new SparseCCMDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
        Bt = new SparseCCMDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
    }

}
