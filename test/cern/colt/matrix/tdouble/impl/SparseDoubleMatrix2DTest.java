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
    
    
    public void testConvertToRCDoubleMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)(Math.random() * NROWS);
            columnindexes[i] = (int)(Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }        
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        RCDoubleMatrix2D B = A.convertToRCDoubleMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToRCMDoubleMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)(Math.random() * NROWS);
            columnindexes[i] = (int)(Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }        
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        RCMDoubleMatrix2D B = A.convertToRCMDoubleMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToCCDoubleMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)(Math.random() * NROWS);
            columnindexes[i] = (int)(Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }        
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        CCDoubleMatrix2D B = A.convertToCCDoubleMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToCCMDoubleMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        double[] values = new double[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)(Math.random() * NROWS);
            columnindexes[i] = (int)(Math.random() * NCOLUMNS);
            values[i] = Math.random();
        }        
        SparseDoubleMatrix2D A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        CCMDoubleMatrix2D B = A.convertToCCMDoubleMatrix2D();
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
        for (int i = 0; i < SIZE; i++){
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