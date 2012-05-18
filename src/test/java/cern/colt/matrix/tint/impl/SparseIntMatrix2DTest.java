package cern.colt.matrix.tint.impl;

import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.IntMatrix2DTest;
import cern.jet.math.tint.IntFunctions;

public class SparseIntMatrix2DTest extends IntMatrix2DTest {

    public SparseIntMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseIntMatrix2D(NROWS, NCOLUMNS);
        B = new SparseIntMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseIntMatrix2D(NCOLUMNS, NROWS);
    }

    public void testGetRowCompressed() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        int[] values = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextInt();
        }
        SparseIntMatrix2D A = new SparseIntMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseRCIntMatrix2D B = A.getRowCompressed(false);
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
        int[] values = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextInt();
        }
        SparseIntMatrix2D S = new SparseIntMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseRCMIntMatrix2D B = S.getRowCompressedModified();
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
        int[] values = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextInt();
        }
        SparseIntMatrix2D S = new SparseIntMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCIntMatrix2D B = S.getColumnCompressed(false);
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
        int[] values = new int[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(rand.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(rand.nextInt() % NCOLUMNS);
            values[i] = rand.nextInt();
        }
        SparseIntMatrix2D S = new SparseIntMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCMIntMatrix2D B = S.getColumnCompressedModified();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testAssignIntArrayIntArrayIntArrayIntIntFunction() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        int[] values = new int[SIZE];
        IntMatrix2D Adense = new DenseIntMatrix2D(A.rows(), A.columns());
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = i % A.rows();
            columnindexes[i] = i % A.columns();
            values[i] = rand.nextInt();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseIntMatrix2D S = new SparseIntMatrix2D(A.rows(), A.columns());
        S.assign(rowindexes, columnindexes, values, IntFunctions.multSecond(2));
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(2 * Adense.getQuick(r, c), S.getQuick(r, c));
            }
        }
    }
}