package cern.colt.function.tfcomplex;

public interface FComplexFComplexFComplexFunction {
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
    abstract public float[] apply(float[] x, float[] y);

}
