package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram;

/**
 * Variable-width axis; A reference implementation of hep.aida.IAxis.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
public class FloatVariableAxis implements FloatIAxis {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected float min;

    protected int bins;

    protected float[] edges;

    /**
     * Constructs and returns an axis with the given bin edges. Example:
     * <tt>edges = (0.2, 1.0, 5.0)</tt> yields an axis with 2 in-range bins
     * <tt>[0.2,1.0), [1.0,5.0)</tt> and 2 extra bins
     * <tt>[-inf,0.2), [5.0,inf]</tt>.
     * 
     * @param edges
     *            the bin boundaries the partition shall have; must be sorted
     *            ascending and must not contain multiple identical elements.
     * @throws IllegalArgumentException
     *             if <tt>edges.length < 1</tt>.
     */
    public FloatVariableAxis(float[] edges) {
        if (edges.length < 1)
            throw new IllegalArgumentException();

        // check if really sorted and has no multiple identical elements
        for (int i = 0; i < edges.length - 1; i++) {
            if (edges[i + 1] <= edges[i]) {
                throw new IllegalArgumentException(
                        "edges must be sorted ascending and must not contain multiple identical values");
            }
        }

        this.min = edges[0];
        this.bins = edges.length - 1;
        this.edges = edges.clone();
    }

    public float binCentre(int index) {
        return (binLowerEdge(index) + binUpperEdge(index)) / 2;
    }

    public float binLowerEdge(int index) {
        if (index == FloatIHistogram.UNDERFLOW)
            return Float.NEGATIVE_INFINITY;
        if (index == FloatIHistogram.OVERFLOW)
            return upperEdge();
        return edges[index];
    }

    public int bins() {
        return bins;
    }

    public float binUpperEdge(int index) {
        if (index == FloatIHistogram.UNDERFLOW)
            return lowerEdge();
        if (index == FloatIHistogram.OVERFLOW)
            return Float.POSITIVE_INFINITY;
        return edges[index + 1];
    }

    public float binWidth(int index) {
        return binUpperEdge(index) - binLowerEdge(index);
    }

    public int coordToIndex(float coord) {
        if (coord < min)
            return FloatIHistogram.UNDERFLOW;

        int index = java.util.Arrays.binarySearch(this.edges, coord);
        // int index = new FloatArrayList(this.edges).binarySearch(coord); //
        // just for debugging
        if (index < 0)
            index = -index - 1 - 1; // not found
        // else index++; // found

        if (index >= bins)
            return FloatIHistogram.OVERFLOW;

        return index;
    }

    public float lowerEdge() {
        return min;
    }

    /**
     * Returns a string representation of the specified array. The string
     * representation consists of a list of the arrays's elements, enclosed in
     * square brackets (<tt>"[]"</tt>). Adjacent elements are separated by the
     * characters <tt>", "</tt> (comma and space).
     * 
     * @return a string representation of the specified array.
     */
    protected static String toString(float[] array) {
        StringBuffer buf = new StringBuffer();
        buf.append("[");
        int maxIndex = array.length - 1;
        for (int i = 0; i <= maxIndex; i++) {
            buf.append(array[i]);
            if (i < maxIndex)
                buf.append(", ");
        }
        buf.append("]");
        return buf.toString();
    }

    public float upperEdge() {
        return edges[edges.length - 1];
    }
}
