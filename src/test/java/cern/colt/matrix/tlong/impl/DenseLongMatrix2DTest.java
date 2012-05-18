package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2DTest;

public class DenseLongMatrix2DTest extends LongMatrix2DTest {

    public DenseLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLongMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLongMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignLongArray() {
        long[] expected = new long[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = rand.nextLong();
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
