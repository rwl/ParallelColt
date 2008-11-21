package cern.colt.matrix.tdcomplex.impl;

import java.util.ArrayList;
import java.util.Random;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDComplexMatrix3D {

	private static Timer t = new Timer();

	private static final String outputFile = "BenchmarkDenseDComplexMatrix3D.txt";

	private static double[][][] a_3d, b_3d;

	private static double[] a_1d, b_1d;

	private static double[] noViewTimes;

	private static double[] viewTimes;

	public static void main(String[] args) {
		org.junit.runner.JUnitCore.main("cern.colt.matrix.impl.BenchmarkDenseDComplexMatrix3D");
	}

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		BenchmarkMatrixKernel.readSettings3D();
		Random rand = new Random(0);
		noViewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
		viewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
		ConcurrencyUtils.setThreadsBeginN_3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
		a_1d = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
		a_3d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
		int idx = 0;
		for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
			for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
				for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
					a_3d[s][r][c] = rand.nextDouble();
					a_1d[idx++] = a_3d[s][r][c];
				}
			}
		}
		b_1d = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
		b_3d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
		idx = 0;
		for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
			for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
				for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
					b_3d[s][r][c] = rand.nextDouble();
					b_1d[idx++] = b_3d[s][r][c];
				}
			}
		}
		BenchmarkMatrixKernel.writePropertiesToFile(outputFile, BenchmarkMatrixKernel.MATRIX_SIZE_3D);
		BenchmarkMatrixKernel.displayProperties(BenchmarkMatrixKernel.MATRIX_SIZE_3D);

	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
		a_1d = null;
		a_3d = null;
		b_1d = null;
		b_3d = null;
		ConcurrencyUtils.resetThreadsBeginN();
		System.gc();
	}

	@Test
    public void testAggregateDComplexDComplexDComplexFunctionDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DComplexDComplexDComplexFunction, DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

  
    @Test
    public void testAggregateDComplexMatrix3DDComplexDComplexDComplexFunctionDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        DComplexMatrix3D B = new DenseDComplexMatrix3D(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DComplexMatrix3D Bv = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DComplexMatrix3D, DComplexDComplexDComplexFunction, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleDouble() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1],BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        double value = Math.random();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(value, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(value, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(value, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(value, value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double, double)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDoubleArray() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_1d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(a_1d);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleArrayArrayArray() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_3d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(a_3d);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double[][][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(DComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(DComplexFunctions.acos);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDComplexMatrix3D() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        DComplexMatrix3D B = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        DComplexMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseDComplexMatrix3D(a_3d);
        DComplexMatrix3D Bv = B.viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.assign(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexMatrix3D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDComplexMatrix3DDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        DComplexMatrix3D B = new DenseDComplexMatrix3D(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(B, DComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DComplexMatrix3D Bv = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 00).assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(Bv, DComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexMatrix3D, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

   
    @Test
    public void testAssignDComplexProcedureDoubleArray() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        double[] value = new double[] {-1, -2};
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexProcedure, double[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDComplexProcedureDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, DComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, DComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, DComplexFunctions.tan);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexProcedure, DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }


    @Test
    public void testCardinality() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            int card = A.cardinality();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                card = A.cardinality();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix3D(a_3d);
        DComplexMatrix3D Av = A.viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            int card = Av.cardinality();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                card = Av.cardinality();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "cardinality()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testFft3() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.fft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.fft3();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2,1,0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.fft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.fft3();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft3()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testFft2Slices() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.fft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.fft2Slices();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.fft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.fft2Slices();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fft2Slices()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetRealPart() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D Bv = Av.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getRealPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getRealPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetImaginaryPart() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D Bv = Av.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getImaginaryPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getImaginaryPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    

    @Test
    public void testGetNonZerosIntArrayListIntArrayListIntArrayListArrayList() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getNonZeros(sliceList, rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
            	rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                A.getNonZeros(sliceList, rowList, colList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(a_3d).viewDice(2, 1, 0);
        sliceList.clear();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.getNonZeros(sliceList, rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();                
            	rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                Av.getNonZeros(sliceList, rowList, colList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNonZeros(IntArrayList, IntArrayList, IntArrayList, ArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft3() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.ifft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.ifft3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.ifft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.ifft3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testIfft2Slices() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.ifft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.ifft2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.ifft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.ifft2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifft2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testSum() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.zSum();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.zSum();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zSum()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testToArray() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][][] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][][] array = Av.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = Av.toArray();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "toArray()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testVectorize() {
        /* No view */
        DComplexMatrix3D A = new DenseDComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix1D B = A.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.vectorize();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix3D Av = new DenseDComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2,1,0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix1D B = Av.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.vectorize();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "vectorize()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
}
