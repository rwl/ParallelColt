package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2DTest;

public class DenseColumnIntMatrix2DTest extends IntMatrix2DTest {

    public DenseColumnIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnIntMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColumnIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColumnIntMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignIntArray() {
        int[] expected = new int[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = Math.max(1, rand.nextInt() % A.rows());
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(expected[idx++], A.getQuick(r, c));
            }
        }
    }

}
