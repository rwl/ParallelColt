package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2DTest;

public class DenseIntMatrix2DTest extends IntMatrix2DTest {

    public DenseIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseIntMatrix2D(NROWS, NCOLUMNS);
        B = new DenseIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseIntMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignIntArray() {
        int[] expected = new int[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = rand.nextInt();
        }
        A.assign(expected);
        int idx = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(0, Math.abs(expected[idx++] - A.getQuick(r, c)));
            }
        }
    }
}
