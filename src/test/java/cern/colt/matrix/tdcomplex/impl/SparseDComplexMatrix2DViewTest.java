package cern.colt.matrix.tdcomplex.impl;

public class SparseDComplexMatrix2DViewTest extends SparseDComplexMatrix2DTest {
    public SparseDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
