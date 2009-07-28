
package cern.colt.matrix.tfloat.impl;

import java.util.Random;

import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import edu.emory.mathcs.utils.IOUtils;

/**
 * Accuracy check of float precision FFT's
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class AccuracyCheckFloatFFT {

    private static int[] sizesPrimes = {997, 4999, 9973, 49999, 99991, 249989 };

    private static int[] sizesPower2 = { 8192, 16384, 32768, 65536, 131072, 262144 };

    private static final int niters = 10;
    
    private static Random r;


    private AccuracyCheckFloatFFT() {

    }

    public static void checkAccuracyRealFFT_1D() {
        System.out.println("Checking accuracy of 1D real forward full FFT (prime sizes)...");
        double[][] errors = new double[niters][sizesPrimes.length];
        for (int i = 0; i < sizesPrimes.length; i++) {
            r = new Random(0);
            for (int j = 0; j < niters; j++) {
                DenseFloatMatrix1D M = new DenseFloatMatrix1D(sizesPrimes[i]);
                fillMatrix_1D(sizesPrimes[i], (float[]) M.elements());
                DenseFComplexMatrix1D Mc = M.getFft();
                Mc.ifft(true);
                DenseFloatMatrix1D Mr = (DenseFloatMatrix1D)Mc.getRealPart();
                errors[j][i] = computeRMSE(M.elements(), Mr.elements());
            }
        }
        IOUtils.writeToFileReal_2D("%g", errors, "pc_rmse_primes_float.txt");        

        System.out.println("Checking accuracy of 1D real forward full FFT (power2 sizes)...");
        errors = new double[niters][sizesPower2.length];
        for (int i = 0; i < sizesPower2.length; i++) {
            r = new Random(0);
            for (int j = 0; j < niters; j++) {
                DenseFloatMatrix1D M = new DenseFloatMatrix1D(sizesPower2[i]);
                fillMatrix_1D(sizesPower2[i], (float[]) M.elements());
                DenseFComplexMatrix1D Mc = M.getFft();
                Mc.ifft(true);
                DenseFloatMatrix1D Mr = (DenseFloatMatrix1D)Mc.getRealPart();
                errors[j][i] = computeRMSE(M.elements(), Mr.elements());
            }
        }
        IOUtils.writeToFileReal_2D("%g", errors, "pc_rmse_power2_float.txt");        

        
    }
    
    private static double computeRMSE(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays are not the same size.");
        }
        float rms = 0;
        float tmp;
        for (int i = 0; i < a.length; i++) {
            tmp = (a[i] - b[i]);
            rms += tmp * tmp;
        }
        return Math.sqrt(rms / (float) a.length);
    }

    public static void main(String[] args) {
        checkAccuracyRealFFT_1D();
        System.exit(0);
    }
    
    private static void fillMatrix_1D(int N, float[] m) {
        for (int i = 0; i < N; i++) {
            m[i] = r.nextFloat();
        }
    }
}
