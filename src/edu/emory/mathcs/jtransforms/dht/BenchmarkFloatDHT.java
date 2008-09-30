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

package edu.emory.mathcs.jtransforms.dht;

import edu.emory.mathcs.utils.ConcurrencyUtils;
import edu.emory.mathcs.utils.IOUtils;

/**
 * Benchmark of single precision DHT's
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class BenchmarkFloatDHT {

    private static int nthread = 2;

    private static int nsize = 6;

    private static int niter = 200;

    private static boolean doWarmup = true;

    private static int initialExponent1D = 17;

    private static int initialExponent2D = 7;

    private static int initialExponent3D = 2;

    private static boolean doScaling = false;

    private BenchmarkFloatDHT() {

    }

    public static void parseArguments(String[] args) {
        for (int i = 0; i < args.length; i++) {
            System.out.println("args[" + i + "]:" + args[i]);
        }

        if ((args == null) || (args.length != 10)) {
            System.out.println("Parameters: <number of threads> <THREADS_BEGIN_N_2D> <THREADS_BEGIN_N_3D> <number of iterations> <perform warm-up> <perform scaling> <number of sizes> <initial exponent for 1D transforms> <initial exponent for 2D transforms> <initial exponent for 3D transforms>");
            System.exit(-1);
        }
        nthread = Integer.parseInt(args[0]);
        ConcurrencyUtils.setThreadsBeginN_2D(Integer.parseInt(args[1]));
        ConcurrencyUtils.setThreadsBeginN_3D(Integer.parseInt(args[2]));
        niter = Integer.parseInt(args[3]);
        doWarmup = Boolean.parseBoolean(args[4]);
        doScaling = Boolean.parseBoolean(args[5]);
        nsize = Integer.parseInt(args[6]);
        initialExponent1D = Integer.parseInt(args[7]);
        initialExponent2D = Integer.parseInt(args[8]);
        initialExponent3D = Integer.parseInt(args[9]);
        ConcurrencyUtils.setNumberOfProcessors(nthread);
    }

    public static void benchmarkForward_1D(int init_exp) {
        int[] sizes = new int[nsize];
        double[] times = new double[nsize];
        float[] x;
        for (int i = 0; i < nsize; i++) {
            int exponent = init_exp + i;
            int N = (int) Math.pow(2, exponent);
            sizes[i] = N;
            System.out.println("Forward DHT 1D of size 2^" + exponent);
            FloatDHT_1D dht = new FloatDHT_1D(N);
            x = new float[N];
            if (doWarmup) { // call the transform twice to warm up
                IOUtils.fillMatrix_1D(N, x);
                dht.forward(x);
                IOUtils.fillMatrix_1D(N, x);
                dht.forward(x);
            }
            float av_time = 0;
            long elapsedTime = 0;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_1D(N, x);
                elapsedTime = System.nanoTime();
                dht.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                av_time = av_time + elapsedTime;
            }
            times[i] = (float) av_time / 1000000.0 / (float) niter;
            System.out.println("Average execution time: " + String.format("%.2f", av_time / 1000000.0 / (float) niter) + " msec");
            x = null;
            dht = null;
            System.gc();
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkFloatForwardDHT_1D.txt", nthread, niter, doWarmup, doScaling, sizes, times);
    }

    public static void benchmarkForward_2D_input_1D(int init_exp) {
        int[] sizes = new int[nsize];
        double[] times = new double[nsize];
        float[] x;
        for (int i = 0; i < nsize; i++) {
            int exponent = init_exp + i;
            int N = (int) Math.pow(2, exponent);
            sizes[i] = N;
            System.out.println("Forward DHT 2D (input 1D) of size 2^" + exponent + " x 2^" + exponent);
            FloatDHT_2D dht2 = new FloatDHT_2D(N, N);
            x = new float[N * N];
            if (doWarmup) { // call the transform twice to warm up
                IOUtils.fillMatrix_2D(N, N, x);
                dht2.forward(x);
                IOUtils.fillMatrix_2D(N, N, x);
                dht2.forward(x);
            }
            float av_time = 0;
            long elapsedTime = 0;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_2D(N, N, x);
                elapsedTime = System.nanoTime();
                dht2.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                av_time = av_time + elapsedTime;
            }
            times[i] = (float) av_time / 1000000.0 / (float) niter;
            System.out.println("Average execution time: " + String.format("%.2f", av_time / 1000000.0 / (float) niter) + " msec");
            x = null;
            dht2 = null;
            System.gc();
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkFloatForwardDHT_2D_input_1D.txt", nthread, niter, doWarmup, doScaling, sizes, times);
    }

    public static void benchmarkForward_2D_input_2D(int init_exp) {
        int[] sizes = new int[nsize];
        double[] times = new double[nsize];
        float[][] x;
        for (int i = 0; i < nsize; i++) {
            int exponent = init_exp + i;
            int N = (int) Math.pow(2, exponent);
            sizes[i] = N;
            System.out.println("Forward DHT 2D (input 2D) of size 2^" + exponent + " x 2^" + exponent);
            FloatDHT_2D dht2 = new FloatDHT_2D(N, N);
            x = new float[N][N];
            if (doWarmup) { // call the transform twice to warm up
                IOUtils.fillMatrix_2D(N, N, x);
                dht2.forward(x);
                IOUtils.fillMatrix_2D(N, N, x);
                dht2.forward(x);
            }
            float av_time = 0;
            long elapsedTime = 0;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_2D(N, N, x);
                elapsedTime = System.nanoTime();
                dht2.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                av_time = av_time + elapsedTime;
            }
            times[i] = (float) av_time / 1000000.0 / (float) niter;
            System.out.println("Average execution time: " + String.format("%.2f", av_time / 1000000.0 / (float) niter) + " msec");
            x = null;
            dht2 = null;
            System.gc();
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkFloatForwardDHT_2D_input_2D.txt", nthread, niter, doWarmup, doScaling, sizes, times);
    }

    public static void benchmarkForward_3D_input_1D(int init_exp) {
        int[] sizes = new int[nsize];
        double[] times = new double[nsize];
        float[] x;
        for (int i = 0; i < nsize; i++) {
            int exponent = init_exp + i;
            int N = (int) Math.pow(2, exponent);
            sizes[i] = N;
            System.out.println("Forward DHT 3D (input 1D) of size 2^" + exponent + " x 2^" + exponent + " x 2^" + exponent);
            FloatDHT_3D dht3 = new FloatDHT_3D(N, N, N);
            x = new float[N * N * N];
            if (doWarmup) { // call the transform twice to warm up
                IOUtils.fillMatrix_3D(N, N, N, x);
                dht3.forward(x);
                IOUtils.fillMatrix_3D(N, N, N, x);
                dht3.forward(x);
            }
            float av_time = 0;
            long elapsedTime = 0;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_3D(N, N, N, x);
                elapsedTime = System.nanoTime();
                dht3.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                av_time = av_time + elapsedTime;
            }
            times[i] = (float) av_time / 1000000.0 / (float) niter;
            System.out.println("Average execution time: " + String.format("%.2f", av_time / 1000000.0 / (float) niter) + " msec");
            x = null;
            dht3 = null;
            System.gc();
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkFloatForwardDHT_3D_input_1D.txt", nthread, niter, doWarmup, doScaling, sizes, times);
    }

    public static void benchmarkForward_3D_input_3D(int init_exp) {
        int[] sizes = new int[nsize];
        double[] times = new double[nsize];
        float[][][] x;
        for (int i = 0; i < nsize; i++) {
            int exponent = init_exp + i;
            int N = (int) Math.pow(2, exponent);
            sizes[i] = N;
            System.out.println("Forward DHT 3D (input 3D) of size 2^" + exponent + " x 2^" + exponent + " x 2^" + exponent);
            FloatDHT_3D dht3 = new FloatDHT_3D(N, N, N);
            x = new float[N][N][N];
            if (doWarmup) { // call the transform twice to warm up
                IOUtils.fillMatrix_3D(N, N, N, x);
                dht3.forward(x);
                IOUtils.fillMatrix_3D(N, N, N, x);
                dht3.forward(x);
            }
            float av_time = 0;
            long elapsedTime = 0;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_3D(N, N, N, x);
                elapsedTime = System.nanoTime();
                dht3.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                av_time = av_time + elapsedTime;
            }
            times[i] = (float) av_time / 1000000.0 / (float) niter;
            System.out.println("Average execution time: " + String.format("%.2f", av_time / 1000000.0 / (float) niter) + " msec");
            x = null;
            dht3 = null;
            System.gc();
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkFloatForwardDHT_3D_input_3D.txt", nthread, niter, doWarmup, doScaling, sizes, times);
    }

    public static void main(String[] args) {
        parseArguments(args);
        benchmarkForward_1D(initialExponent1D);
        benchmarkForward_2D_input_1D(initialExponent2D);
        benchmarkForward_2D_input_2D(initialExponent2D);
        benchmarkForward_3D_input_1D(initialExponent3D);
        benchmarkForward_3D_input_3D(initialExponent3D);
        System.exit(0);

    }
}
