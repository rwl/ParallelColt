
package cern.colt.matrix.tdouble.impl;

import java.util.Random;

import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import edu.emory.mathcs.utils.IOUtils;

/**
 * Accuracy check of double precision FFT's
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class AccuracyCheckDoubleFFT {

    private static int[] sizesPrimes = {997, 4999, 9973, 49999, 99991, 249989 };

    private static int[] sizesPower2 = { 8192, 16384, 32768, 65536, 131072, 262144};

    private static final int niters = 10;
    
    private static Random r;


    private AccuracyCheckDoubleFFT() {

    }

    public static void checkAccuracyRealFFT_1D() {
        System.out.println("Checking accuracy of 1D real forward full FFT (prime sizes)...");
        double[][] errors = new double[niters][sizesPrimes.length];
        for (int i = 0; i < sizesPrimes.length; i++) {
            r = new Random(0);
            for (int j = 0; j < niters; j++) {
                DenseDoubleMatrix1D M = new DenseDoubleMatrix1D(sizesPrimes[i]);
                fillMatrix_1D(sizesPrimes[i], (double[]) M.elements());
                DenseDComplexMatrix1D Mc = M.getFft();
                Mc.ifft(true);
                DenseDoubleMatrix1D Mr = (DenseDoubleMatrix1D)Mc.getRealPart();
                errors[j][i] = computeRMSE(M.elements(), Mr.elements());
            }
        }
        IOUtils.writeToFileReal_2D("%g", errors, "pc_rmse_primes_double.txt");        

        System.out.println("Checking accuracy of 1D real forward full FFT (power2 sizes)...");
        errors = new double[niters][sizesPower2.length];
        for (int i = 0; i < sizesPower2.length; i++) {
            r = new Random(0);
            for (int j = 0; j < niters; j++) {
                DenseDoubleMatrix1D M = new DenseDoubleMatrix1D(sizesPower2[i]);
                fillMatrix_1D(sizesPower2[i], (double[]) M.elements());
                DenseDComplexMatrix1D Mc = M.getFft();
                Mc.ifft(true);
                DenseDoubleMatrix1D Mr = (DenseDoubleMatrix1D)Mc.getRealPart();
                errors[j][i] = computeRMSE(M.elements(), Mr.elements());
            }
        }
        IOUtils.writeToFileReal_2D("%g", errors, "pc_rmse_power2_double.txt");        

        
    }
    
    private static double computeRMSE(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays are not the same size.");
        }
        double rms = 0;
        double tmp;
        for (int i = 0; i < a.length; i++) {
            tmp = (a[i] - b[i]);
            rms += tmp * tmp;
        }
        return Math.sqrt(rms / (double) a.length);
    }

    public static void main(String[] args) {
        checkAccuracyRealFFT_1D();
        System.exit(0);
    }
    
    private static void fillMatrix_1D(int N, double[] m) {
        for (int i = 0; i < N; i++) {
            m[i] = r.nextDouble();
        }
    }
}
