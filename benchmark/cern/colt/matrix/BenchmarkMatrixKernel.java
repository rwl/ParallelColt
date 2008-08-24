/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.Date;

/**
 * Not yet documented.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class BenchmarkMatrixKernel {

    public static int MATRIX_SIZE_1D = (int) Math.pow(2, 19);

    public static int[] MATRIX_SIZE_2D = new int[] { (int) Math.pow(2, 10), (int) Math.pow(2, 10) };

    public static int[] MATRIX_SIZE_3D = new int[] { (int) Math.pow(2, 7), (int) Math.pow(2, 7), (int) Math.pow(2, 7) };

    public static int[] NTHREADS = new int[] { 1, 2 };

    public static int NITERS = 100;

    private static String settingsFileName1D = "settings1D.txt";

    private static String settingsFileName2D = "settings2D.txt";

    private static String settingsFileName3D = "settings3D.txt";

    /**
     * Benchmark constructor comment.
     */
    private BenchmarkMatrixKernel() {
    }

    public static void readSettings1D() {
        settingsFileName1D = System.getProperty("settingsFile1D", settingsFileName1D);
        File settingsFile = new File(settingsFileName1D);
        if (settingsFile.exists()) {
            try {
                RandomAccessFile input = null;
                input = new RandomAccessFile(settingsFileName1D, "r");
                String line;
                line = input.readLine();
                line = input.readLine();
                String[] stringThreads = line.split(",");
                NTHREADS = new int[stringThreads.length];
                for (int i = 0; i < stringThreads.length; i++) {
                    NTHREADS[i] = Integer.parseInt(stringThreads[i].trim());
                }

                line = input.readLine();
                MATRIX_SIZE_1D = Integer.parseInt(line.trim());

                line = input.readLine();
                NITERS = Integer.parseInt(line.trim());

                input.close();

                System.out.println("Settings were loaded");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Settings file not found. Default settings will be used.");
            System.out.println("NTHREADS = " + Arrays.toString(NTHREADS));
            System.out.println("MATRIX_SIZE = " + MATRIX_SIZE_1D);
            System.out.println("NITERS = " + NITERS);
        }
    }

    public static void readSettings2D() {
        settingsFileName2D = System.getProperty("settingsFile2D", settingsFileName2D);
        File settingsFile = new File(settingsFileName2D);
        if (settingsFile.exists()) {
            try {
                RandomAccessFile input = null;
                input = new RandomAccessFile(settingsFileName2D, "r");
                String line;
                line = input.readLine();
                line = input.readLine();
                String[] stringThreads = line.split(",");
                NTHREADS = new int[stringThreads.length];
                for (int i = 0; i < stringThreads.length; i++) {
                    NTHREADS[i] = Integer.parseInt(stringThreads[i].trim());
                }

                line = input.readLine();
                MATRIX_SIZE_2D[0] = Integer.parseInt(line.trim());
                line = input.readLine();
                MATRIX_SIZE_2D[1] = Integer.parseInt(line.trim());

                line = input.readLine();
                NITERS = Integer.parseInt(line.trim());

                input.close();

                System.out.println("Settings were loaded");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Settings file not found. Default settings will be used.");
            System.out.println("NTHREADS = " + Arrays.toString(NTHREADS));
            System.out.println("MATRIX_SIZE = " + MATRIX_SIZE_2D[0] + " x " + MATRIX_SIZE_2D[1]);
            System.out.println("NITERS = " + NITERS);
        }
    }

    public static void readSettings3D() {
        settingsFileName3D = System.getProperty("settingsFile3D", settingsFileName3D);
        File settingsFile = new File(settingsFileName3D);
        if (settingsFile.exists()) {
            try {
                RandomAccessFile input = null;
                input = new RandomAccessFile(settingsFileName3D, "r");
                String line;
                line = input.readLine();
                line = input.readLine();
                String[] stringThreads = line.split(",");
                NTHREADS = new int[stringThreads.length];
                for (int i = 0; i < stringThreads.length; i++) {
                    NTHREADS[i] = Integer.parseInt(stringThreads[i].trim());
                }

                line = input.readLine();
                MATRIX_SIZE_3D[0] = Integer.parseInt(line.trim());
                line = input.readLine();
                MATRIX_SIZE_3D[1] = Integer.parseInt(line.trim());
                line = input.readLine();
                MATRIX_SIZE_3D[2] = Integer.parseInt(line.trim());

                line = input.readLine();
                NITERS = Integer.parseInt(line.trim());

                input.close();

                System.out.println("Settings were loaded");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Settings file not found. Default settings will be used.");
            System.out.println("NTHREADS = " + Arrays.toString(NTHREADS));
            System.out.println("MATRIX_SIZE = " + MATRIX_SIZE_3D[0] + " x " + MATRIX_SIZE_3D[1] + " x " + MATRIX_SIZE_3D[2]);
            System.out.println("NITERS = " + NITERS);
        }
    }

    /**
     * Saves properties in a file.
     * 
     * @param filename
     * @param size
     */
    public static void writePropertiesToFile(String filename, int[] size) {
        String[] properties = { "os.name", "os.version", "os.arch", "java.vendor", "java.version" };
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename, false));
            out.write(new Date().toString());
            out.newLine();
            out.write("System properties:");
            out.newLine();
            out.write("\tos.name = " + System.getProperty(properties[0]));
            out.newLine();
            out.write("\tos.version = " + System.getProperty(properties[1]));
            out.newLine();
            out.write("\tos.arch = " + System.getProperty(properties[2]));
            out.newLine();
            out.write("\tjava.vendor = " + System.getProperty(properties[3]));
            out.newLine();
            out.write("\tjava.version = " + System.getProperty(properties[4]));
            out.newLine();
            out.write("\tavailable processors = " + Runtime.getRuntime().availableProcessors());
            out.newLine();
            switch (size.length) {
            case 1:
                out.write("Matrix size = " + size[0]);
                break;
            case 2:
                out.write("Matrix size = " + size[0] + " x " + size[1]);
                break;
            case 3:
                out.write("Matrix size = " + size[0] + " x " + size[1] + " x " + size[2]);
                break;
            }
            out.newLine();
            out.write("--------------------------------------------------------------------------------------------------");
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves results of the benchmark in a file.
     * 
     * @param filename
     * @param method
     * @param threads
     * @param noViewTimes
     * @param viewTimes
     */
    public static void writeMatrixBenchmarkResultsToFile(String filename, String method, int[] threads, double[] noViewTimes, double[] viewTimes) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename, true));
            out.write("Method = " + method);
            out.newLine();
            out.write("\tNo view execution times:");
            out.newLine();
            for (int i = 0; i < threads.length; i++) {
                if (threads[i] == 1) {
                    out.write("\t\t" + threads[i] + " thread  = " + String.format("%.2f", noViewTimes[i]) + " milliseconds");
                } else {
                    out.write("\t\t" + threads[i] + " threads = " + String.format("%.2f", noViewTimes[i]) + " milliseconds");
                }
                out.newLine();
            }
            out.write("\tView execution times:");
            out.newLine();
            for (int i = 0; i < threads.length; i++) {
                if (threads[i] == 1) {
                    out.write("\t\t" + threads[i] + " thread  = " + String.format("%.2f", viewTimes[i]) + " milliseconds");
                } else {
                    out.write("\t\t" + threads[i] + " threads = " + String.format("%.2f", viewTimes[i]) + " milliseconds");
                }
                out.newLine();
            }
            out.write("--------------------------------------------------------------------------------------------------");
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Displays properties.
     * @param size
     */
    public static void displayProperties(int[] size) {
        String[] properties = { "os.name", "os.version", "os.arch", "java.vendor", "java.version" };
        System.out.println(new Date().toString());
        System.out.println("System properties:");
        System.out.println("\tos.name = " + System.getProperty(properties[0]));
        System.out.println("\tos.version = " + System.getProperty(properties[1]));
        System.out.println("\tos.arch = " + System.getProperty(properties[2]));
        System.out.println("\tjava.vendor = " + System.getProperty(properties[3]));
        System.out.println("\tjava.version = " + System.getProperty(properties[4]));
        System.out.println("\tavailable processors = " + Runtime.getRuntime().availableProcessors());
        switch (size.length) {
        case 1:
            System.out.println("Matrix size = " + size[0]);
            break;
        case 2:
            System.out.println("Matrix size = " + size[0] + " x " + size[1]);
            break;
        case 3:
            System.out.println("Matrix size = " + size[0] + " x " + size[1] + " x " + size[2]);
            break;
        }
        System.out.println("--------------------------------------------------------------------------------------------------");
    }

    /**
     * Displays results of the benchmark
     * 
     * @param method
     * @param threads
     * @param noViewTimes
     * @param viewTimes
     * @param append
     */
    public static void displayMatrixBenchmarkResults(String method, int[] threads, double[] noViewTimes, double[] viewTimes) {
        System.out.println("Method = " + method);
        System.out.println("\tNo view execution times:");
        for (int i = 0; i < threads.length; i++) {
            if (threads[i] == 1) {
                System.out.println("\t\t" + threads[i] + " thread  = " + String.format("%.2f", noViewTimes[i]) + " milliseconds");
            } else {
                System.out.println("\t\t" + threads[i] + " threads = " + String.format("%.2f", noViewTimes[i]) + " milliseconds");
            }
        }
        System.out.println("\tView execution times:");
        for (int i = 0; i < threads.length; i++) {
            if (threads[i] == 1) {
                System.out.println("\t\t" + threads[i] + " thread  = " + String.format("%.2f", viewTimes[i]) + " milliseconds");
            } else {
                System.out.println("\t\t" + threads[i] + " threads = " + String.format("%.2f", viewTimes[i]) + " milliseconds");
            }
        }
        System.out.println("--------------------------------------------------------------------------------------------------");
    }

    /**
     * Executes procedure repeatedly until more than minSeconds have elapsed.
     */
    public static float run(double minSeconds, TimerProcedure procedure) {
        long iter = 0;
        long minMillis = (long) (minSeconds * 1000);
        long begin = System.currentTimeMillis();
        long limit = begin + minMillis;
        while (System.currentTimeMillis() < limit) {
            procedure.init();
            procedure.apply(null);
            iter++;
        }
        long end = System.currentTimeMillis();
        if (minSeconds / iter < 0.1) {
            // unreliable timing due to very fast iteration;
            // reading, starting and stopping timer distorts measurement
            // do it again with minimal timer overhead
            // System.out.println("iter="+iter+",
            // minSeconds/iter="+minSeconds/iter);
            begin = System.currentTimeMillis();
            for (long i = iter; --i >= 0;) {
                procedure.init();
                procedure.apply(null);
            }
            end = System.currentTimeMillis();
        }

        long begin2 = System.currentTimeMillis();
        int dummy = 1; // prevent compiler from optimizing away the loop
        for (long i = iter; --i >= 0;) {
            dummy *= i;
            procedure.init();
        }
        long end2 = System.currentTimeMillis();
        long elapsed = (end - begin) - (end2 - begin2);
        // if (dummy != 0) throw new RuntimeException("dummy != 0");

        return (float) elapsed / 1000.0f / iter;
    }

    /**
     * Returns a String with the system's properties (vendor, version, operating
     * system, etc.)
     */
    public static String systemInfo() {
        String[] properties = { "java.vm.vendor", "java.vm.version", "java.vm.name", "os.name", "os.version", "os.arch", "java.version", "java.vendor", "java.vendor.url"
        /*
         * "java.vm.specification.version", "java.vm.specification.vendor",
         * "java.vm.specification.name", "java.specification.version",
         * "java.specification.vendor", "java.specification.name"
         */
        };

        // build string matrix
        cern.colt.matrix.ObjectMatrix2D matrix = new cern.colt.matrix.impl.DenseObjectMatrix2D(properties.length, 2);
        matrix.viewColumn(0).assign(properties);

        // retrieve property values
        for (int i = 0; i < properties.length; i++) {
            String value = System.getProperty(properties[i]);
            if (value == null)
                value = "?"; // prop not available
            matrix.set(i, 1, value);
        }

        // format matrix
        cern.colt.matrix.objectalgo.ObjectFormatter formatter = new cern.colt.matrix.objectalgo.ObjectFormatter();
        formatter.setPrintShape(false);
        return formatter.toString(matrix);
    }
}
