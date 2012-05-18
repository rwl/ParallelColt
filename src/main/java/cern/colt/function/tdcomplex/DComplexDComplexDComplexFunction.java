package cern.colt.function.tdcomplex;

public interface DComplexDComplexDComplexFunction {
    /**
     * Applies a function to two complex arguments.
     * 
     * @param x
     *            the first argument passed to the function.
     * @param y
     *            the second argument passed to the function.
     * 
     * @return the result of the function.
     */
    abstract public double[] apply(double[] x, double[] y);

}
