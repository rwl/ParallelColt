package hep.aida.tfloat;

/**
 * An IAxis represents a binned histogram axis. A 1D Histogram would have one
 * Axis representing the X axis, while a 2D Histogram would have two axes
 * representing the X and Y Axis.
 * 
 * @author Pavel Binko, Dino Ferrero Merlino, Wolfgang Hoschek, Tony Johnson,
 *         Andreas Pfeiffer, and others.
 * @version 1.0, 23/03/2000
 */
public interface FloatIAxis extends java.io.Serializable {
    static final long serialVersionUID = 1020;

    /**
     * Centre of the bin specified.
     * 
     * @param index
     *            Bin number (0...bins()-1) or OVERFLOW or UNDERFLOW.
     */
    public float binCentre(int index);

    /**
     * Lower edge of the specified bin.
     * 
     * @param index
     *            Bin number (0...bins()-1) or OVERFLOW or UNDERFLOW.
     * @return the lower edge of the bin; for the underflow bin this is
     *         <tt>Float.NEGATIVE_INFINITY</tt>.
     */
    public float binLowerEdge(int index);

    /**
     * The number of bins (excluding underflow and overflow) on the axis.
     */
    public int bins();

    /**
     * Upper edge of the specified bin.
     * 
     * @param index
     *            Bin number (0...bins()-1) or OVERFLOW or UNDERFLOW.
     * @return the upper edge of the bin; for the overflow bin this is
     *         <tt>Float.POSITIVE_INFINITY</tt>.
     */
    public float binUpperEdge(int index);

    /**
     * Width of the bin specified.
     * 
     * @param index
     *            Bin number (0...bins()-1) or OVERFLOW or UNDERFLOW.
     */
    public float binWidth(int index);

    /**
     * Converts a coordinate on the axis to a bin number. If the coordinate is <
     * lowerEdge returns UNDERFLOW, and if the coordinate is >= upperEdge
     * returns OVERFLOW.
     */
    public int coordToIndex(float coord);

    /**
     * Lower axis edge.
     */
    public float lowerEdge();

    /**
     * Upper axis edge.
     */
    public float upperEdge();
}
