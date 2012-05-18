package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram;

/**
 * Fixed-width axis; A reference implementation of hep.aida.IAxis.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
public class FloatFixedAxis implements FloatIAxis {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private int bins;

    private float min;

    private float binWidth;

    // Package private for ease of use in Histogram1D and Histogram2D
    private int xunder, xover;

    /**
     * Create an Axis
     * 
     * @param bins
     *            Number of bins
     * @param min
     *            Minimum for axis
     * @param max
     *            Maximum for axis
     */
    public FloatFixedAxis(int bins, float min, float max) {
        if (bins < 1)
            throw new IllegalArgumentException("bins=" + bins);
        if (max <= min)
            throw new IllegalArgumentException("max <= min");

        // Note, for internal consistency we save only min and binWidth
        // and always use these quantities for all calculations. Due to
        // rounding errors the return value from upperEdge is not necessarily
        // exactly equal to max

        this.bins = bins;
        this.min = min;
        this.binWidth = (max - min) / bins;

        // our internal definition of overflow/underflow differs from
        // that of the outside world
        // this.under = 0;
        // this.over = bins+1;

    }

    public float binCentre(int index) {
        return min + binWidth * index + binWidth / 2;
    }

    public float binLowerEdge(int index) {
        if (index == FloatIHistogram.UNDERFLOW)
            return Float.NEGATIVE_INFINITY;
        if (index == FloatIHistogram.OVERFLOW)
            return upperEdge();
        return min + binWidth * index;
    }

    public int bins() {
        return bins;
    }

    public float binUpperEdge(int index) {
        if (index == FloatIHistogram.UNDERFLOW)
            return min;
        if (index == FloatIHistogram.OVERFLOW)
            return Float.POSITIVE_INFINITY;
        return min + binWidth * (index + 1);
    }

    public float binWidth(int index) {
        return binWidth;
    }

    public int coordToIndex(float coord) {
        if (coord < min)
            return FloatIHistogram.UNDERFLOW;
        int index = (int) Math.floor((coord - min) / binWidth);
        if (index >= bins)
            return FloatIHistogram.OVERFLOW;

        return index;
    }

    public float lowerEdge() {
        return min;
    }

    public float upperEdge() {
        return min + binWidth * bins;
    }

    /**
     * This package private method is similar to coordToIndex except that it
     * returns our internal definition for overflow/underflow
     */
    int xgetBin(float coord) {
        if (coord < min)
            return xunder;
        int index = (int) Math.floor((coord - min) / binWidth);
        if (index > bins)
            return xover;
        return index + 1;
    }

    /**
     * Package private method to map from the external representation of bin
     * number to our internal representation of bin number
     */
    int xmap(int index) {
        if (index >= bins)
            throw new IllegalArgumentException("bin=" + index);
        if (index >= 0)
            return index + 1;
        if (index == FloatIHistogram.UNDERFLOW)
            return xunder;
        if (index == FloatIHistogram.OVERFLOW)
            return xover;
        throw new IllegalArgumentException("bin=" + index);
    }
}
