package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram;
import hep.aida.tfloat.FloatIHistogram1D;

/**
 * Abstract base class extracting and implementing most of the redundancy of the
 * interface.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
abstract class FloatAbstractHistogram1D extends FloatHistogram implements FloatIHistogram1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    protected FloatIAxis xAxis;

    FloatAbstractHistogram1D(String title) {
        super(title);
    }

    public int allEntries() {
        return entries() + extraEntries();
    }

    public int dimensions() {
        return 1;
    }

    public int entries() {
        int entries = 0;
        for (int i = xAxis.bins(); --i >= 0;)
            entries += binEntries(i);
        return entries;
    }

    public int extraEntries() {
        // return entries[xAxis.under] + entries[xAxis.over];
        return binEntries(UNDERFLOW) + binEntries(OVERFLOW);
    }

    /**
     * Package private method to map from the external representation of bin
     * number to our internal representation of bin number
     */
    int map(int index) {
        int bins = xAxis.bins() + 2;
        if (index >= bins)
            throw new IllegalArgumentException("bin=" + index);
        if (index >= 0)
            return index + 1;
        if (index == FloatIHistogram.UNDERFLOW)
            return 0;
        if (index == FloatIHistogram.OVERFLOW)
            return bins - 1;
        throw new IllegalArgumentException("bin=" + index);
    }

    public int[] minMaxBins() {
        float minValue = Float.MAX_VALUE;
        float maxValue = Float.MIN_VALUE;
        int minBinX = -1;
        int maxBinX = -1;
        for (int i = xAxis.bins(); --i >= 0;) {
            float value = binHeight(i);
            if (value < minValue) {
                minValue = value;
                minBinX = i;
            }
            if (value > maxValue) {
                maxValue = value;
                maxBinX = i;
            }
        }
        int[] result = { minBinX, maxBinX };
        return result;
    }

    public float sumAllBinHeights() {
        return sumBinHeights() + sumExtraBinHeights();
    }

    public float sumBinHeights() {
        float sum = 0;
        for (int i = xAxis.bins(); --i >= 0;)
            sum += binHeight(i);
        return sum;
    }

    public float sumExtraBinHeights() {
        return binHeight(UNDERFLOW) + binHeight(OVERFLOW);
        // return heights[xAxis.under] + heights[xAxis.over];
    }

    public FloatIAxis xAxis() {
        return xAxis;
    }
}
