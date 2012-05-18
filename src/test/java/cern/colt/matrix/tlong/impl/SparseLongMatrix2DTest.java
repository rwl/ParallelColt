package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix2DTest;
import cern.jet.math.tlong.LongFunctions;

public class SparseLongMatrix2DTest extends LongMatrix2DTest {

    public SparseLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseLongMatrix2D(NROWS, NCOLUMNS);
        B = new SparseLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseLongMatrix2D(NCOLUMNS, NROWS);
    }

    public void testGetRowCompressed() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        long[] values = new long[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextLong();
        }
        SparseLongMatrix2D A = new SparseLongMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseRCLongMatrix2D B = A.getRowCompressed(false);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
        B = A.getRowCompressed(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testGetRowCompressedModified() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        long[] values = new long[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextLong();
        }
        SparseLongMatrix2D S = new SparseLongMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseRCMLongMatrix2D B = S.getRowCompressedModified();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testGetColumnCompressed() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        long[] values = new long[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextLong();
        }
        SparseLongMatrix2D S = new SparseLongMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCLongMatrix2D B = S.getColumnCompressed(false);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
        B = S.getColumnCompressed(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }

    }

    public void testGetColumnCompressedModified() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        long[] values = new long[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextLong();
        }
        SparseLongMatrix2D S = new SparseLongMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCMLongMatrix2D B = S.getColumnCompressedModified();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testAssignIntArrayIntArrayLongArrayLongLongFunction() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        long[] values = new long[SIZE];
        LongMatrix2D Adense = new DenseLongMatrix2D(A.rows(), A.columns());
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = i % A.rows();
            columnindexes[i] = i % A.columns();
            values[i] = rand.nextLong();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseLongMatrix2D S = new SparseLongMatrix2D(A.rows(), A.columns());
        S.assign(rowindexes, columnindexes, values, LongFunctions.multSecond(2));
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(2 * Adense.getQuick(r, c), S.getQuick(r, c));
            }
        }
    }
}