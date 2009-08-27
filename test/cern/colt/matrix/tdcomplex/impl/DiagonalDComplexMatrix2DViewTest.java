package cern.colt.matrix.tdcomplex.impl;

public class DiagonalDComplexMatrix2DViewTest extends DiagonalDComplexMatrix2DTest {

    public DiagonalDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalDComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalDComplexMatrix2D) A).diagonalLength();
        A = A.viewDice();
        B = new DiagonalDComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX).viewDice();
        Bt = new DiagonalDComplexMatrix2D(NROWS, NCOLUMNS, DINDEX).viewDice();
    }

}
