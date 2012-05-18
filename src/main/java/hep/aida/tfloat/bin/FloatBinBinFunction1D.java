package hep.aida.tfloat.bin;

/**
 * Interface that represents a function object: a function that takes two bins
 * as arguments and returns a single value.
 */
public interface FloatBinBinFunction1D {
    /**
     * Applies a function to two bin arguments.
     * 
     * @param x
     *            the first argument passed to the function.
     * @param y
     *            the second argument passed to the function.
     * @return the result of the function.
     */
    abstract public float apply(DynamicFloatBin1D x, DynamicFloatBin1D y);
}
