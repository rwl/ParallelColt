/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is JTransforms.
 *
 * The Initial Developer of the Original Code is
 * Piotr Wendykier, Emory University.
 * Portions created by the Initial Developer are Copyright (C) 2007
 * the Initial Developer. All Rights Reserved.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jtransforms.fft;

import edu.emory.mathcs.utils.IOUtils;

/**
 * Accuracy check of single precision FFT's
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class AccuracyCheckFloatFFT {

    private AccuracyCheckFloatFFT() {

    }

    public static void checkAccuracyComplexFFT_1D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 1D complex FFT...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_1D fft = new FloatFFT_1D(size);
            float e, err = 0.0f;
            float[] a = new float[2 * size];
            IOUtils.fillMatrix_1D(2 * size, a);
            float[] b = new float[2 * size];
            IOUtils.fillMatrix_1D(2 * size, b);
            fft.complexForward(a);
            fft.complexInverse(a, true);
            for (int j = 0; j < 2 * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft = null;
            System.gc();
        }
    }

    public static void checkAccuracyComplexFFT_2D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 2D complex FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, 2 * size, a);
            float[] b = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, 2 * size, b);
            fft2.complexForward(a);
            fft2.complexInverse(a, true);
            for (int j = 0; j < 2 * size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D complex FFT (float[][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[][] a = new float[size][2 * size];
            IOUtils.fillMatrix_2D(size, 2 * size, a);
            float[][] b = new float[size][2 * size];
            IOUtils.fillMatrix_2D(size, 2 * size, b);
            fft2.complexForward(a);
            fft2.complexInverse(a, true);
            for (int r = 0; r < size; r++) {
                for (int c = 0; c < 2 * size; c++) {
                    e = Math.abs(b[r][c] - a[r][c]);
                    err = Math.max(err, e);
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }

    }

    public static void checkAccuracyComplexFFT_3D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 3D complex FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, 2 * size, a);
            float[] b = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, 2 * size, b);
            fft3.complexForward(a);
            fft3.complexInverse(a, true);
            for (int j = 0; j < 2 * size * size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D complex FFT (float[][][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[][][] a = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, 2 * size, a);
            float[][][] b = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, 2 * size, b);
            fft3.complexForward(a);
            fft3.complexInverse(a, true);
            for (int s = 0; s < size; s++) {
                for (int r = 0; r < size; r++) {
                    for (int c = 0; c < 2 * size; c++) {
                        e = Math.abs(b[s][r][c] - a[s][r][c]);
                        err = Math.max(err, e);
                    }
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
    }

    public static void checkAccuracyRealFFT_1D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 1D real FFT...");

        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_1D fft = new FloatFFT_1D(size);
            float e, err = 0.0f;
            float[] a = new float[size];
            IOUtils.fillMatrix_1D(size, a);
            float[] b = new float[size];
            IOUtils.fillMatrix_1D(size, b);
            fft.realForward(b);
            fft.realInverse(b, true);
            for (int j = 0; j < size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft = null;
            System.gc();
        }
        System.out.println("Checking accuracy of on 1D real forward full FFT...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_1D fft = new FloatFFT_1D(size);
            float e, err = 0.0f;
            float[] a = new float[2 * size];
            IOUtils.fillMatrix_1D(size, a);
            float[] b = new float[2 * size];
            IOUtils.fillMatrix_1D(size, b);
            fft.realForwardFull(b);
            fft.complexInverse(b, true);
            for (int j = 0; j < size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 1D real inverse full FFT...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_1D fft = new FloatFFT_1D(size);
            float e, err = 0.0f;
            float[] a = new float[2 * size];
            IOUtils.fillMatrix_1D(size, a);
            float[] b = new float[2 * size];
            IOUtils.fillMatrix_1D(size, b);
            fft.realInverseFull(b, true);
            fft.complexForward(b);
            for (int j = 0; j < size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft = null;
            System.gc();
        }

    }

    public static void checkAccuracyRealFFT_2D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 2D real FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[] a = new float[size * size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[] b = new float[size * size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realForward(b);
            fft2.realInverse(b, true);
            for (int j = 0; j < size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D real FFT (float[][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[][] a = new float[size][size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[][] b = new float[size][size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realForward(b);
            fft2.realInverse(b, true);
            for (int r = 0; r < size; r++) {
                for (int c = 0; c < size; c++) {
                    e = Math.abs(b[r][c] - a[r][c]);
                    err = Math.max(err, e);
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D real forward full FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[] b = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realForwardFull(b);
            fft2.complexInverse(b, true);
            for (int j = 0; j < size * size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D real forward full FFT (float[][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[][] a = new float[size][size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[][] b = new float[size][2 * size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realForwardFull(b);
            fft2.complexInverse(b, true);
            for (int r = 0; r < size; r++) {
                for (int c = 0; c < size; c++) {
                    e = Math.abs(b[r][2 * c] - a[r][c]);
                    err = Math.max(err, e);
                    e = Math.abs(b[r][2 * c + 1]);
                    err = Math.max(err, e);
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D real inverse full FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[] b = new float[2 * size * size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realInverseFull(b, true);
            fft2.complexForward(b);
            for (int j = 0; j < size * size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D real inverse full FFT (float[][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_2D fft2 = new FloatFFT_2D(size, size);
            float e, err = 0.0f;
            float[][] a = new float[size][size];
            IOUtils.fillMatrix_2D(size, size, a);
            float[][] b = new float[size][2 * size];
            IOUtils.fillMatrix_2D(size, size, b);
            fft2.realInverseFull(b, true);
            fft2.complexForward(b);
            for (int r = 0; r < size; r++) {
                for (int c = 0; c < size; c++) {
                    e = Math.abs(b[r][2 * c] - a[r][c]);
                    err = Math.max(err, e);
                    e = Math.abs(b[r][2 * c + 1]);
                    err = Math.max(err, e);
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            fft2 = null;
            System.gc();
        }

    }

    public static void checkAccuracyRealFFT_3D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 3D real FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[] a = new float[size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[] b = new float[size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realForward(b);
            fft3.realInverse(b, true);
            for (int j = 0; j < size * size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D real FFT (float[][][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[][][] a = new float[size][size][size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[][][] b = new float[size][size][size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realForward(b);
            fft3.realInverse(b, true);
            for (int s = 0; s < size; s++) {
                for (int r = 0; r < size; r++) {
                    for (int c = 0; c < size; c++) {
                        e = Math.abs(b[s][r][c] - a[s][r][c]);
                        err = Math.max(err, e);
                    }
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D real forward full FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[] b = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realForwardFull(b);
            fft3.complexInverse(b, true);
            for (int j = 0; j < size * size * size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D real forward full FFT (float[][][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[][][] a = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[][][] b = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realForwardFull(b);
            fft3.complexInverse(b, true);
            for (int s = 0; s < size; s++) {
                for (int r = 0; r < size; r++) {
                    for (int c = 0; c < size; c++) {
                        e = Math.abs(b[s][r][2 * c] - a[s][r][c]);
                        err = Math.max(err, e);
                        e = Math.abs(b[s][r][2 * c + 1]);
                        err = Math.max(err, e);
                    }
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D real inverse full FFT (float[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[] a = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[] b = new float[2 * size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realInverseFull(b, true);
            fft3.complexForward(b);
            for (int j = 0; j < size * size * size; j++) {
                e = Math.abs(b[2 * j] - a[j]);
                err = Math.max(err, e);
                e = Math.abs(b[2 * j + 1]);
                err = Math.max(err, e);
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 3D real inverse full FFT (float[][][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            FloatFFT_3D fft3 = new FloatFFT_3D(size, size, size);
            float e, err = 0.0f;
            float[][][] a = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            float[][][] b = new float[size][size][2 * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            fft3.realInverseFull(b, true);
            fft3.complexForward(b);
            for (int s = 0; s < size; s++) {
                for (int r = 0; r < size; r++) {
                    for (int c = 0; c < size; c++) {
                        e = Math.abs(b[s][r][2 * c] - a[s][r][c]);
                        err = Math.max(err, e);
                        e = Math.abs(b[s][r][2 * c + 1]);
                        err = Math.max(err, e);
                    }
                }
            }
            if (err > 1e-5) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            fft3 = null;
            System.gc();
        }
    }

    public static void main(String[] args) {
        checkAccuracyComplexFFT_1D(0, 21);
        checkAccuracyRealFFT_1D(0, 21);
        checkAccuracyComplexFFT_2D(1, 11);
        checkAccuracyRealFFT_2D(1, 11);
        checkAccuracyComplexFFT_3D(1, 7);
        checkAccuracyRealFFT_3D(1, 7);
        System.exit(0);
    }
}
