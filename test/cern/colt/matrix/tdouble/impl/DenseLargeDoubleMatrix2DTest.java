package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.DenseLargeDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class DenseLargeDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public DenseLargeDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLargeDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLargeDoubleMatrix2D(NCOLUMNS, NROWS);
    }

    public void testDct2() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dct2(true);
        ((WrapperDoubleMatrix2D) A).idct2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dctColumns(true);
        ((WrapperDoubleMatrix2D) A).idctColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dctRows(true);
        ((WrapperDoubleMatrix2D) A).idctRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDht2() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dht2();
        ((WrapperDoubleMatrix2D) A).idht2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dhtColumns();
        ((WrapperDoubleMatrix2D) A).idhtColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dhtRows();
        ((WrapperDoubleMatrix2D) A).idhtRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDst2() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dst2(true);
        ((WrapperDoubleMatrix2D) A).idst2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstColumns() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dstColumns(true);
        ((WrapperDoubleMatrix2D) A).idstColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstRows() {
        DoubleMatrix2D Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).dstRows(true);
        ((WrapperDoubleMatrix2D) A).idstRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testFft2() {
        int nrows = 64;
        int ncolumns = 128;
        DoubleMatrix2D A = new DenseLargeDoubleMatrix2D(nrows, ncolumns);
        DoubleMatrix2D Acopy = A.copy();
        ((DenseLargeDoubleMatrix2D) A).fft2();
        ((DenseLargeDoubleMatrix2D) A).ifft2(true);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncolumns; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }

        A = A.viewDice();
        Acopy = A.copy();
        ((WrapperDoubleMatrix2D) A).fft2();
        ((WrapperDoubleMatrix2D) A).ifft2(true);
        for (int r = 0; r < ncolumns; r++) {
            for (int c = 0; c < nrows; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetFft2() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getFft2();
        ((DenseLargeDComplexMatrix2D) Ac).ifft2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfft2() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getIfft2(true);
        ((DenseLargeDComplexMatrix2D) Ac).fft2();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftColumns() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getFftColumns();
        ((DenseLargeDComplexMatrix2D) Ac).ifftColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftColumns() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getIfftColumns(true);
        ((DenseLargeDComplexMatrix2D) Ac).fftColumns();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftRows() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getFftRows();
        ((DenseLargeDComplexMatrix2D) Ac).ifftRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftRows() {
        DoubleMatrix2D Acopy = A.copy();
        DComplexMatrix2D Ac = ((WrapperDoubleMatrix2D) A).getIfftRows(true);
        ((DenseLargeDComplexMatrix2D) Ac).fftRows();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }
}
