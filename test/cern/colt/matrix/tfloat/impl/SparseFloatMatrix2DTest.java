package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2DTest;
import cern.jet.math.tfloat.FloatFunctions;

public class SparseFloatMatrix2DTest extends FloatMatrix2DTest {

    public SparseFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseFloatMatrix2D(NROWS, NCOLUMNS);
        B = new SparseFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseFloatMatrix2D(NCOLUMNS, NROWS);
    }
    
    
    public void testConvertToRCFloatMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)(Math.random() * NROWS);
            columnindexes[i] = (int)(Math.random() * NCOLUMNS);
            values[i] = (float)Math.random();
        }        
        SparseFloatMatrix2D A = new SparseFloatMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        RCFloatMatrix2D B = A.convertToRCFloatMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToRCMFloatMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)((float)Math.random() * NROWS);
            columnindexes[i] = (int)((float)Math.random() * NCOLUMNS);
            values[i] = (float)Math.random();
        }        
        SparseFloatMatrix2D A = new SparseFloatMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        RCMFloatMatrix2D B = A.convertToRCMFloatMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToCCFloatMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)((float)Math.random() * NROWS);
            columnindexes[i] = (int)((float)Math.random() * NCOLUMNS);
            values[i] = (float)Math.random();
        }        
        SparseFloatMatrix2D A = new SparseFloatMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        CCFloatMatrix2D B = A.convertToCCFloatMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testConvertToCCMFloatMatrix2D() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = (int)((float)Math.random() * NROWS);
            columnindexes[i] = (int)((float)Math.random() * NCOLUMNS);
            values[i] = (float)Math.random();
        }        
        SparseFloatMatrix2D A = new SparseFloatMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);
        CCMFloatMatrix2D B = A.convertToCCMFloatMatrix2D();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(r, c));
            }            
        }        
    }
    
    public void testAssignIntArrayIntArrayFloatArrayFloatFloatFunction() {
        int SIZE = NROWS * NCOLUMNS;
        int[] rowindexes = new int[SIZE];
        int[] columnindexes = new int[SIZE];
        float[] values = new float[SIZE];
        FloatMatrix2D Adense = new DenseFloatMatrix2D(NROWS, NCOLUMNS);
        for (int i = 0; i < SIZE; i++){
            rowindexes[i] = i % NROWS;
            columnindexes[i] = i % NCOLUMNS;
            values[i] = (float)Math.random();
            Adense.setQuick(rowindexes[i], columnindexes[i], values[i]);
        }
        SparseFloatMatrix2D A = new SparseFloatMatrix2D(NROWS, NCOLUMNS, rowindexes, columnindexes, values);       
        A.assign(rowindexes, columnindexes, values, FloatFunctions.plus);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(2 * Adense.getQuick(r, c), A.getQuick(r, c));
            }            
        }          
    }
}