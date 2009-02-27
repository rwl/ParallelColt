package cern.colt.function.tfcomplex;

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

    /**
     * Applies a function to a complex argument.
     * 
     * @param re
     *            real part of an argument passed to the function
     * @param im
     *            imaginary part of an argument passed to the function
     * @return the result of the function.
     */
    abstract public float[] apply(float re, float im);
}
