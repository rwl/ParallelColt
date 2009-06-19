package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram;
import hep.aida.tfloat.FloatIHistogram1D;
import hep.aida.tfloat.FloatIHistogram2D;

/**
 * Abstract base class extracting and implementing most of the redundancy of the
 * interface.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
abstract class FloatAbstractHistogram2D extends FloatHistogram implements FloatIHistogram2D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    protected FloatIAxis xAxis, yAxis;

    FloatAbstractHistogram2D(String title) {
        super(title);
    }

    public int allEntries() {
        int n = 0;
        for (int i = xAxis.bins(); --i >= -2;)
            for (int j = yAxis.bins(); --j >= -2;) {
                n += binEntries(i, j);
            }
        return n;
    }

    public int binEntriesX(int indexX) {
        return projectionX().binEntries(indexX);
    }

    public int binEntriesY(int indexY) {
        return projectionY().binEntries(indexY);
    }

    public float binHeightX(int indexX) {
        return projectionX().binHeight(indexX);
    }

    public float binHeightY(int indexY) {
        return projectionY().binHeight(indexY);
    }

    public int dimensions() {
        return 2;
    }

    public int entries() {
        int n = 0;
        for (int i = 0; i < xAxis.bins(); i++)
            for (int j = 0; j < yAxis.bins(); j++) {
                n += binEntries(i, j);
            }
        return n;
    }

    public int extraEntries() {
        return allEntries() - entries();
    }

    public void fill(float x, float y) {
        fill(x, y, 1);
    }

    /**
     * The precise meaning of the arguments to the public slice methods is
     * somewhat ambiguous, so we define this internal slice method and clearly
     * specify its arguments.
     * <p>
     * <b>Note 0</b>indexY1 and indexY2 use our INTERNAL bin numbering scheme
     * <b>Note 1</b>The slice is done between indexY1 and indexY2 INCLUSIVE
     * <b>Note 2</b>indexY1 and indexY2 may include the use of under and over
     * flow bins <b>Note 3</b>There is no note 3 (yet)
     */
    protected abstract FloatIHistogram1D internalSliceX(String title, int indexY1, int indexY2);

    /**
     * The precise meaning of the arguments to the public slice methods is
     * somewhat ambiguous, so we define this internal slice method and clearly
     * specify its arguments.
     * <p>
     * <b>Note 0</b>indexX1 and indexX2 use our INTERNAL bin numbering scheme
     * <b>Note 1</b>The slice is done between indexX1 and indexX2 INCLUSIVE
     * <b>Note 2</b>indexX1 and indexX2 may include the use of under and over
     * flow bins <b>Note 3</b>There is no note 3 (yet)
     */
    protected abstract FloatIHistogram1D internalSliceY(String title, int indexX1, int indexX2);

    /**
     * Package private method to map from the external representation of bin
     * number to our internal representation of bin number
     */
    int mapX(int index) {
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

    /**
     * Package private method to map from the external representation of bin
     * number to our internal representation of bin number
     */
    int mapY(int index) {
        int bins = yAxis.bins() + 2;
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
        int minBinY = -1;
        int maxBinX = -1;
        int maxBinY = -1;
        for (int i = xAxis.bins(); --i >= 0;) {
            for (int j = yAxis.bins(); --j >= 0;) {
                float value = binHeight(i, j);
                if (value < minValue) {
                    minValue = value;
                    minBinX = i;
                    minBinY = j;
                }
                if (value > maxValue) {
                    maxValue = value;
                    maxBinX = i;
                    maxBinY = j;
                }
            }
        }
        int[] result = { minBinX, minBinY, maxBinX, maxBinY };
        return result;
    }

    public FloatIHistogram1D projectionX() {
        String newTitle = title() + " (projectionX)";
        // return internalSliceX(newTitle,yAxis.under,yAxis.over);
        return internalSliceX(newTitle, mapY(FloatIHistogram.UNDERFLOW), mapY(FloatIHistogram.OVERFLOW));
    }

    public FloatIHistogram1D projectionY() {
        String newTitle = title() + " (projectionY)";
        // return internalSliceY(newTitle,xAxis.under,xAxis.over);
        return internalSliceY(newTitle, mapX(FloatIHistogram.UNDERFLOW), mapX(FloatIHistogram.OVERFLOW));
    }

    public FloatIHistogram1D sliceX(int indexY) {
        // int start = yAxis.map(indexY);
        int start = mapY(indexY);
        String newTitle = title() + " (sliceX [" + indexY + "])";
        return internalSliceX(newTitle, start, start);
    }

    public FloatIHistogram1D sliceX(int indexY1, int indexY2) {
        // int start = yAxis.map(indexY1);
        // int stop = yAxis.map(indexY2);
        int start = mapY(indexY1);
        int stop = mapY(indexY2);
        String newTitle = title() + " (sliceX [" + indexY1 + ":" + indexY2 + "])";
        return internalSliceX(newTitle, start, stop);
    }

    public FloatIHistogram1D sliceY(int indexX) {
        // int start = xAxis.map(indexX);
        int start = mapX(indexX);
        String newTitle = title() + " (sliceY [" + indexX + "])";
        return internalSliceY(newTitle, start, start);
    }

    public FloatIHistogram1D sliceY(int indexX1, int indexX2) {
        // int start = xAxis.map(indexX1);
        // int stop = xAxis.map(indexX2);
        int start = mapX(indexX1);
        int stop = mapX(indexX2);
        String newTitle = title() + " (slicey [" + indexX1 + ":" + indexX2 + "])";
        return internalSliceY(newTitle, start, stop);
    }

    public float sumAllBinHeights() {
        float n = 0;
        for (int i = xAxis.bins(); --i >= -2;)
            for (int j = yAxis.bins(); --j >= -2;) {
                n += binHeight(i, j);
            }
        return n;
    }

    public float sumBinHeights() {
        float n = 0;
        for (int i = 0; i < xAxis.bins(); i++)
            for (int j = 0; j < yAxis.bins(); j++) {
                n += binHeight(i, j);
            }
        return n;
    }

    public float sumExtraBinHeights() {
        return sumAllBinHeights() - sumBinHeights();
    }

    public FloatIAxis xAxis() {
        return xAxis;
    }

    public FloatIAxis yAxis() {
        return yAxis;
    }
}
