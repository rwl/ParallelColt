package cern.colt.matrix.tdcomplex.algo;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.jet.math.tdcomplex.DComplexFunctions;
import cern.jet.math.tdouble.DoubleFunctions;

/**
 * Linear algebraic matrix operations operating on dense complex matrices.
 * 
 * @author Wolfgang Hoschek (wolfgang.hoschek@cern.ch)
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * @author Richard Lincoln (r.w.lincoln@gmail.com)
 */
public class DenseDComplexAlgebra extends cern.colt.PersistentObject {
    private static final long serialVersionUID = 1L;

    /**
     * A default Algebra object; has {@link DComplexProperty#DEFAULT} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final DenseDComplexAlgebra DEFAULT;

    /**
     * A default Algebra object; has {@link DComplexProperty#ZERO} attached for
     * tolerance. Allows ommiting to construct an Algebra object time and again.
     * 
     * Note that this Algebra object is immutable. Any attempt to assign a new
     * Property object to it (via method <tt>setProperty</tt>), or to alter the
     * tolerance of its property object (via
     * <tt>property().setTolerance(...)</tt>) will throw an exception.
     */
    public static final DenseDComplexAlgebra ZERO;

    /**
     * The property object attached to this instance.
     */
    protected DComplexProperty property;

    static {
        // don't use new Algebra(Property.DEFAULT.tolerance()), because then
        // property object would be mutable.
        DEFAULT = new DenseDComplexAlgebra();
        DEFAULT.property = DComplexProperty.DEFAULT; // immutable property object

        ZERO = new DenseDComplexAlgebra();
        ZERO.property = DComplexProperty.ZERO; // immutable property object
    }

    /**
     * Constructs a new instance with an equality tolerance given by
     * <tt>Property.DEFAULT.tolerance()</tt>.
     */
    public DenseDComplexAlgebra() {
        this(DComplexProperty.DEFAULT.tolerance());
    }

    /**
     * Constructs a new instance with the given equality tolerance.
     * 
     * @param tolerance
     *            the tolerance to be used for equality operations.
     */
    public DenseDComplexAlgebra(double tolerance) {
        setProperty(new DComplexProperty(tolerance));
    }

    /**
     * Attaches the given property object to this Algebra, defining tolerance.
     * 
     * @param property
     *            the Property object to be attached.
     * @throws UnsupportedOperationException
     *             if <tt>this==DEFAULT && property!=this.property()</tt> - The
     *             DEFAULT Algebra object is immutable.
     * @throws UnsupportedOperationException
     *             if <tt>this==ZERO && property!=this.property()</tt> - The
     *             ZERO Algebra object is immutable.
     * @see #property
     */
    public void setProperty(DComplexProperty property) {
        if (this == DEFAULT && property != this.property)
            throw new IllegalArgumentException("Attempted to modify immutable object.");
        if (this == ZERO && property != this.property)
            throw new IllegalArgumentException("Attempted to modify immutable object.");
        this.property = property;
    }

    /**
     * Returns the infinity norm of vector <tt>x</tt>, which is
     * <tt>Max(abs(x[i]))</tt>.
     */
    public double normInfinity(DComplexMatrix1D x) {
        if (x.size() == 0)
            return 0;
        DoubleMatrix1D d = x.assign(DComplexFunctions.abs).getRealPart();
        return d.aggregate(DoubleFunctions.max, DoubleFunctions.identity);
    }

}
