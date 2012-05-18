package cern.colt.matrix.tfcomplex.impl;

public class DiagonalFComplexMatrix2DViewTest extends DiagonalFComplexMatrix2DTest {

    public DiagonalFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalFComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalFComplexMatrix2D) A).diagonalLength();
        A = A.viewDice();
        B = new DiagonalFComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX).viewDice();
        Bt = new DiagonalFComplexMatrix2D(NROWS, NCOLUMNS, DINDEX).viewDice();
    }

}
