package cern.colt.function;

public interface FComplexFComplexFunction {
    /**
     * Applies a function to a complex argument.
     * 
     * @param x
     *            an argument passed to the function.
     * 
     * @return the result of the function.
     */
    abstract public float[] apply(float[] x);
}
