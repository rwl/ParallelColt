package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class DenseDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public DenseDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new DenseDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseDoubleMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignDoubleArray() {
        double[] expected = new double[A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(0, Math.abs(expected[idx++] - A.getQuick(r, c)), TOL);
            }
        }

    }

    public void testAssignFloatArray() {
        float[] expected = new float[NROWS * NCOLUMNS];
        for (int i = 0; i < NROWS * NCOLUMNS; i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[idx++], A.getQuick(r, c), TOL);
            }
        }
    }

    public void testDct2() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dct2(true);
        ((DenseDoubleMatrix2D) A).idct2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dctColumns(true);
        ((DenseDoubleMatrix2D) A).idctColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dctRows(true);
        ((DenseDoubleMatrix2D) A).idctRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDht2() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dht2();
        ((DenseDoubleMatrix2D) A).idht2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dhtColumns();
        ((DenseDoubleMatrix2D) A).idhtColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dhtRows();
        ((DenseDoubleMatrix2D) A).idhtRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDst2() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dst2(true);
        ((DenseDoubleMatrix2D) A).idst2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dstColumns(true);
        ((DenseDoubleMatrix2D) A).idstColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).dstRows(true);
        ((DenseDoubleMatrix2D) A).idstRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testFft2() {
    	int nrows = 64;
    	int ncolumns = 128; 
        DoubleMatrix2D A = new DenseDoubleMatrix2D(nrows, ncolumns);
        DoubleMatrix2D Acopy = A.copy();
        ((DenseDoubleMatrix2D) A).fft2();
        ((DenseDoubleMatrix2D) A).ifft2(true);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncolumns; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetFft2() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getFft2();
        ((DenseDComplexMatrix2D) Ac).ifft2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfft2() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getIfft2(true);
        ((DenseDComplexMatrix2D) Ac).fft2();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftColumns() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getFftColumns();
        ((DenseDComplexMatrix2D) Ac).ifftColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftColumns() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getIfftColumns(true);
        ((DenseDComplexMatrix2D) Ac).fftColumns();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftRows() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getFftRows();
        ((DenseDComplexMatrix2D) Ac).ifftRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftRows() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((DenseDoubleMatrix2D) A).getIfftRows(true);
        ((DenseDComplexMatrix2D) Ac).fftRows();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

}
