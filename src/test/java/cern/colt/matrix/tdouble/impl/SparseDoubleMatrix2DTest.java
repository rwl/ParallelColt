package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2DTest;
import cern.jet.math.tdouble.DoubleFunctions;

public class SparseDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public SparseDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new SparseDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseDoubleMatrix2D(NCOLUMNS, NROWS);
    }

    public void testGetRowCompressed() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseRCDoubleMatrix2D B = A.getRowCompressed(false);
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
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D S = new SparseDoubleMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseRCMDoubleMatrix2D B = S.getRowCompressedModified();
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
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D S = new SparseDoubleMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCDoubleMatrix2D B = S.getColumnCompressed(false);
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
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D S = new SparseDoubleMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCMDoubleMatrix2D B = S.getColumnCompressedModified();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testAssignIntArrayIntArrayDoubleArrayDoubleDoubleFunction() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        DoubleMatrix2D Adense = new DenseDoubleMatrix2D(A.rows(), A.columns());
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = i % A.rows();
            columnindexes[i] = i % A.columns();
            values[i] = Math.random();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseDoubleMatrix2D S = new SparseDoubleMatrix2D(A.rows(), A.columns());
        S.assign(rowindexes, columnindexes, values, DoubleFunctions.multSecond(2));
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(2 * Adense.getQuick(r, c), S.getQuick(r, c));
            }
        }
    }
}