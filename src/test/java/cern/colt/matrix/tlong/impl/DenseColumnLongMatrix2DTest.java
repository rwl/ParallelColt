package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2DTest;

public class DenseColumnLongMatrix2DTest extends LongMatrix2DTest {

    public DenseColumnLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnLongMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColumnLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColumnLongMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignLongArray() {
        long[] expected = new long[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = Math.max(1, rand.nextLong() % A.rows());
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
