package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2DTest;
import cern.jet.math.tfloat.FloatFunctions;

public class SparseFloatMatrix2DTest extends FloatMatrix2DTest {

    public SparseFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix2D(NROWS, NCOLUMNS);
        B = new SparseFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseFloatMatrix2D(NCOLUMNS, NROWS);
    }

    public void testGetRowCompressed() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = (float) Math.random();
        }
        SparseFloatMatrix2D S = new SparseFloatMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseRCFloatMatrix2D B = S.getRowCompressed(false);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
        B = S.getRowCompressed(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testGetRowCompressedModified() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = (float) Math.random();
        }
        SparseFloatMatrix2D S = new SparseFloatMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseRCMFloatMatrix2D B = S.getRowCompressedModified();
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
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = (float) Math.random();
        }
        SparseFloatMatrix2D S = new SparseFloatMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCFloatMatrix2D B = S.getColumnCompressed(false);
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
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = (int) Math.abs(random.nextInt() % NROWS);
            columnindexes[i] = (int) Math.abs(random.nextInt() % NCOLUMNS);
            values[i] = (float) Math.random();
        }
        SparseFloatMatrix2D S = new SparseFloatMatrix2D(A.rows(), A.columns(), rowindexes, columnindexes, values);
        SparseCCMFloatMatrix2D B = S.getColumnCompressedModified();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(S.getQuick(r, c), B.getQuick(r, c));
            }
        }
    }

    public void testAssignIntArrayIntArrayFloatArrayFloatFloatFunction() {
        int SIZE = A.rows() * A.columns();
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        FloatMatrix2D Adense = new DenseFloatMatrix2D(A.rows(), A.columns());
        for (int i = 0; i < SIZE; i++) {
            rowindexes[i] = i % A.rows();
            columnindexes[i] = i % A.columns();
            values[i] = (float) Math.random();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseFloatMatrix2D S = new SparseFloatMatrix2D(A.rows(), A.columns());
        S.assign(rowindexes, columnindexes, values, FloatFunctions.plusMultSecond(2));
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(2 * Adense.getQuick(r, c), S.getQuick(r, c));
            }
        }
    }
}
