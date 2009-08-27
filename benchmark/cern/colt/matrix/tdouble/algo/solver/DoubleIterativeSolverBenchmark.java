package cern.colt.matrix.tdouble.algo.solver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;

import junit.framework.TestCase;
import cern.colt.Timer;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.solver.preconditioner.DoublePreconditioner;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * Benchmark of double precision iterative solver
 */
public abstract class DoubleIterativeSolverBenchmark extends TestCase {

    /**
     * Number of times to repeat tests
     */
    protected int repeat;

    protected int[] nthreads;

    protected int maxIter;

    /**
     * Square system matrix
     */
    protected DoubleMatrix2D A;

    /**
     * Right hand side and the solution vector
     */
    protected DoubleMatrix1D b, x;

    /**
     * Iterative solver to use
     */
    protected DoubleIterativeSolver solver;

    /**
     * Preconditioner to use
     */
    protected DoublePreconditioner M;

    protected DoubleIterationMonitor monitor;

    protected static final Timer t = new Timer();

    /**
     * Constructor for IterativeSolverTest
     */
    public DoubleIterativeSolverBenchmark(String arg0) {
        super(arg0);
    }

    protected void setUp() throws Exception {
        readIterativeSolverBenchmarkSettings();

        int n = A.rows();
        x = new DenseDoubleMatrix1D(n);
        b = new DenseDoubleMatrix1D(n);
        // Create solver and preconditioner
        createSolver();
        solver.setIterationMonitor(monitor);
        M.setMatrix(A);
        solver.setPreconditioner(M);
        x.assign(1);
        // Compute the correct right hand sides
        b = A.zMult(x, b);
        x = new DenseDoubleMatrix1D(n);
    }

    protected void tearDown() throws Exception {
        b = x = null;
        solver = null;
    }

    protected abstract void createSolver() throws Exception;

    public void testBenchmark() {
        System.out.println("Benchmarking " + solver.getClass().getName() + " with preconditioner "
                + solver.getPreconditioner().getClass().getName());
        for (int k = 0; k < nthreads.length; k++) {
            System.out.println("\tNumber of threads = " + nthreads[k]);
            ConcurrencyUtils.setNumberOfThreads(nthreads[k]);
            IterativeSolverDoubleNotConvergedException ex = null;
            double elapsedTime = 0;
            for (int i = 0; i < repeat; ++i) {
                t.reset().start();
                try {
                    solver.solve(A, b, x);
                } catch (IterativeSolverDoubleNotConvergedException e) {
                    ex = e;
                }
                t.stop();
                elapsedTime += t.seconds();
                x.assign(0);
            }
            if (ex != null) {
                System.out.println("\t\tSolver did not converge: " + ex.getReason() + ". Residual=" + ex.getResidual());
            }
            System.out.println("\t\tNumber of iterations performed = " + solver.getIterationMonitor().iterations());
            System.out.println("\t\tAverage execution time = " + (elapsedTime / repeat) + " seconds");
        }
    }

    private void readIterativeSolverBenchmarkSettings() {
        String settingsPath = System.getProperty("iterativeSolverSettingsFile");
        if (settingsPath == null) {
            throw new IllegalArgumentException("Property iterativeSolverSettingsFile not found!");
        }
        File settingsFile = new File(settingsPath);
        if (settingsFile.exists()) {
            try {
                RandomAccessFile input = null;
                input = new RandomAccessFile(settingsPath, "r");
                String line;
                line = input.readLine();
                line = input.readLine();
                String matrixPath = line;
                MatrixVectorReader reader = new MatrixVectorReader(new BufferedReader(new FileReader(matrixPath)));
                SparseDoubleMatrix2D Aco = new SparseDoubleMatrix2D(reader);
                A = Aco.getRowCompressed(true);
                line = input.readLine();
                repeat = Integer.parseInt(line.trim());
                line = input.readLine();
                maxIter = Integer.parseInt(line.trim());
                monitor = new DefaultDoubleIterationMonitor(maxIter, 0, 0, Double.MAX_VALUE);
                line = input.readLine();
                String[] stringThreads = line.split(",");
                nthreads = new int[stringThreads.length];
                for (int i = 0; i < stringThreads.length; i++) {
                    nthreads[i] = Integer.parseInt(stringThreads[i].trim());
                }

                input.close();
                System.out.println("Settings were loaded");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            throw new IllegalArgumentException("The settings file does not exist!");
        }
    }

}
