package cern.colt.function.tdcomplex;

public interface DComplexDComplexFunction {
    /**
     * Applies a function to a complex argument.
     * 
     * @param x
     *            an argument passed to the function.
     * 
     * @return the result of the function.
     */
    abstract public double[] apply(double[] x);
}
