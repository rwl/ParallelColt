package hep.aida.tfloat.bin;

/**
 * Interface that represents a function object: a function that takes two bins
 * as arguments and returns a single value.
 */
public interface FloatBinFunction1D {
    /**
     * Applies a function to one bin argument.
     * 
     * @param x
     *            the argument passed to the function.
     * @return the result of the function.
     */
    abstract public float apply(DynamicFloatBin1D x);

    /**
     * Returns the name of this function.
     * 
     * @return the name of this function.
     */
    abstract public String name();
}
