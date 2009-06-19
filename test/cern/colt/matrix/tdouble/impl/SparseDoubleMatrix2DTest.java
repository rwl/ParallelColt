package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2DTest;
import cern.jet.math.tdouble.DoubleFunctions;

public class SparseDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public SparseDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
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
            rowindexes[i] = (int) (Math.random() * NROWS);
            columnindexes[i] = (int) (Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseRCDoubleMatrix2D B = A.getRowCompressed(false);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
        B = A.getRowCompressed(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testGetRowCompressedModified() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) (Math.random() * NROWS);
            columnindexes[i] = (int) (Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseRCMDoubleMatrix2D B = A.getRowCompressedModified();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testGetColumnCompressed() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) (Math.random() * NROWS);
            columnindexes[i] = (int) (Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseCCDoubleMatrix2D B = A.getColumnCompressed(false);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
        B = A.getColumnCompressed(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }

    }

    public void testGetColumnCompressedModified() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) (Math.random() * NROWS);
            columnindexes[i] = (int) (Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        SparseCCMDoubleMatrix2D B = A.getColumnCompressedModified();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testAssignIntArrayIntArrayDoubleArrayDoubleDoubleFunction() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        DoubleMatrix2D Adense = new DenseDoubleMatrix2D(NROWS, NCOLUMNS);
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = i % NROWS;
            columnindexes[i] = i % NCOLUMNS;
            values[i] = Math.random();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        A.assign(rowindexes, columnindexes, values, DoubleFunctions.plus);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(2 * Adense.getQuick(r, c), A.getQuick(r, c));
            }
        }
    }
}