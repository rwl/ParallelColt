package cern.colt.function.tdcomplex;

/**
 * Interface that represents a procedure object: a procedure that takes a single
 * argument and does not return a value.
 */
public interface DComplexProcedure {
    /**
     * Applies a procedure to an argument. Optionally can return a boolean flag
     * to inform the object calling the procedure.
     * 
     * <p>
     * Example: forEach() methods often use procedure objects. To signal to a
     * forEach() method whether iteration should continue normally or terminate
     * (because for example a matching element has been found), a procedure can
     * return <tt>false</tt> to indicate termination and <tt>true</tt> to
     * indicate continuation.
     * 
     * @param element
     *            element passed to the procedure.
     * @return a flag to inform the object calling the procedure.
     */
    abstract public boolean apply(double[] element);
}