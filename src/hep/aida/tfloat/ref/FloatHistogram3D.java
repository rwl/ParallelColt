package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram2D;
import hep.aida.tfloat.FloatIHistogram3D;

/**
 * A reference implementation of hep.aida.IHistogram3D. The goal is to provide a
 * clear implementation rather than the most efficient implementation. However,
 * performance seems fine - filling 3 * 10^5 points/sec, both using FixedAxis or
 * VariableAxis.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
public class FloatHistogram3D extends FloatAbstractHistogram3D implements FloatIHistogram3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private float[][][] heights;

    private float[][][] errors;

    private int[][][] entries;

    private int nEntry; // total number of times fill called

    private float sumWeight; // Sum of all weights

    private float sumWeightSquared; // Sum of the squares of the weights

    private float meanX, rmsX;

    private float meanY, rmsY;

    private float meanZ, rmsZ;

    /**
     * Creates a variable-width histogram. Example:
     * <tt>xEdges = (0.2, 1.0, 5.0, 6.0), yEdges = (-5, 0, 7), zEdges = (-5, 0, 7)</tt>
     * yields 3*2*2 in-range bins.
     * 
     * @param title
     *            The histogram title.
     * @param xEdges
     *            the bin boundaries the x-axis shall have; must be sorted
     *            ascending and must not contain multiple identical elements.
     * @param yEdges
     *            the bin boundaries the y-axis shall have; must be sorted
     *            ascending and must not contain multiple identical elements.
     * @param zEdges
     *            the bin boundaries the z-axis shall have; must be sorted
     *            ascending and must not contain multiple identical elements.
     * @throws IllegalArgumentException
     *             if
     *             <tt>xEdges.length < 1 || yEdges.length < 1|| zEdges.length < 1</tt>
     *             .
     */
    public FloatHistogram3D(String title, float[] xEdges, float[] yEdges, float[] zEdges) {
        this(title, new FloatVariableAxis(xEdges), new FloatVariableAxis(yEdges), new FloatVariableAxis(zEdges));
    }

    /**
     * Creates a fixed-width histogram.
     * 
     * @param title
     *            The histogram title.
     * @param xBins
     *            The number of bins on the X axis.
     * @param xMin
     *            The minimum value on the X axis.
     * @param xMax
     *            The maximum value on the X axis.
     * @param yBins
     *            The number of bins on the Y axis.
     * @param yMin
     *            The minimum value on the Y axis.
     * @param yMax
     *            The maximum value on the Y axis.
     * @param zBins
     *            The number of bins on the Z axis.
     * @param zMin
     *            The minimum value on the Z axis.
     * @param zMax
     *            The maximum value on the Z axis.
     */
    public FloatHistogram3D(String title, int xBins, float xMin, float xMax, int yBins, float yMin, float yMax,
            int zBins, float zMin, float zMax) {
        this(title, new FloatFixedAxis(xBins, xMin, xMax), new FloatFixedAxis(yBins, yMin, yMax), new FloatFixedAxis(
                zBins, zMin, zMax));
    }

    /**
     * Creates a histogram with the given axis binning.
     * 
     * @param title
     *            The histogram title.
     * @param xAxis
     *            The x-axis description to be used for binning.
     * @param yAxis
     *            The y-axis description to be used for binning.
     * @param zAxis
     *            The z-axis description to be used for binning.
     */
    public FloatHistogram3D(String title, FloatIAxis xAxis, FloatIAxis yAxis, FloatIAxis zAxis) {
        super(title);
        this.xAxis = xAxis;
        this.yAxis = yAxis;
        this.zAxis = zAxis;
        int xBins = xAxis.bins();
        int yBins = yAxis.bins();
        int zBins = zAxis.bins();

        entries = new int[xBins + 2][yBins + 2][zBins + 2];
        heights = new float[xBins + 2][yBins + 2][zBins + 2];
        errors = new float[xBins + 2][yBins + 2][zBins + 2];

    }

    public int allEntries() {
        return nEntry;
    }

    public int binEntries(int indexX, int indexY, int indexZ) {
        return entries[mapX(indexX)][mapY(indexY)][mapZ(indexZ)];
    }

    public float binError(int indexX, int indexY, int indexZ) {
        return (float) Math.sqrt(errors[mapX(indexX)][mapY(indexY)][mapZ(indexZ)]);
    }

    public float binHeight(int indexX, int indexY, int indexZ) {
        return heights[mapX(indexX)][mapY(indexY)][mapZ(indexZ)];
    }

    public float equivalentBinEntries() {
        return sumWeight * sumWeight / sumWeightSquared;
    }

    public void fill(float x, float y, float z) {
        int xBin = mapX(xAxis.coordToIndex(x));
        int yBin = mapY(yAxis.coordToIndex(y));
        int zBin = mapZ(zAxis.coordToIndex(z));
        entries[xBin][yBin][zBin]++;
        heights[xBin][yBin][zBin]++;
        errors[xBin][yBin][zBin]++;
        nEntry++;
        sumWeight++;
        sumWeightSquared++;
        meanX += x;
        rmsX += x;
        meanY += y;
        rmsY += y;
        meanZ += z;
        rmsZ += z;
    }

    public void fill(float x, float y, float z, float weight) {
        int xBin = mapX(xAxis.coordToIndex(x));
        int yBin = mapY(yAxis.coordToIndex(y));
        int zBin = mapZ(zAxis.coordToIndex(z));
        entries[xBin][yBin][zBin]++;
        heights[xBin][yBin][zBin] += weight;
        errors[xBin][yBin][zBin] += weight * weight;
        nEntry++;
        sumWeight += weight;
        sumWeightSquared += weight * weight;
        meanX += x * weight;
        rmsX += x * weight * weight;
        meanY += y * weight;
        rmsY += y * weight * weight;
        meanZ += z * weight;
        rmsZ += z * weight * weight;
    }

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

    protected FloatIHistogram2D internalSliceXY(String title, int indexZ1, int indexZ2) {
        // Attention: our internal definition of bins has been choosen
        // so that this works properly even if the indeces passed in include
        // the underflow or overflow bins
        if (indexZ2 < indexZ1)
            throw new IllegalArgumentException("Invalid bin range");

        int xBins = xAxis.bins() + 2;
        int yBins = yAxis.bins() + 2;
        int[][] sliceEntries = new int[xBins][yBins];
        float[][] sliceHeights = new float[xBins][yBins];
        float[][] sliceErrors = new float[xBins][yBins];

        for (int i = 0; i < xBins; i++) {
            for (int j = 0; j < yBins; j++) {
                for (int k = indexZ1; k <= indexZ2; k++) {
                    sliceEntries[i][j] += entries[i][j][k];
                    sliceHeights[i][j] += heights[i][j][k];
                    sliceErrors[i][j] += errors[i][j][k];
                }
            }
        }
        FloatHistogram2D result = new FloatHistogram2D(title, xAxis, yAxis);
        result.setContents(sliceEntries, sliceHeights, sliceErrors);
        return result;
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

    protected FloatIHistogram2D internalSliceXZ(String title, int indexY1, int indexY2) {
        // Attention: our internal definition of bins has been choosen
        // so that this works properly even if the indeces passed in include
        // the underflow or overflow bins
        if (indexY2 < indexY1)
            throw new IllegalArgumentException("Invalid bin range");

        int xBins = xAxis.bins() + 2;
        int zBins = zAxis.bins() + 2;
        int[][] sliceEntries = new int[xBins][zBins];
        float[][] sliceHeights = new float[xBins][zBins];
        float[][] sliceErrors = new float[xBins][zBins];

        for (int i = 0; i < xBins; i++) {
            for (int j = indexY1; j <= indexY2; j++) {
                for (int k = 0; i < zBins; k++) {
                    sliceEntries[i][k] += entries[i][j][k];
                    sliceHeights[i][k] += heights[i][j][k];
                    sliceErrors[i][k] += errors[i][j][k];
                }
            }
        }
        FloatHistogram2D result = new FloatHistogram2D(title, xAxis, zAxis);
        result.setContents(sliceEntries, sliceHeights, sliceErrors);
        return result;
    }

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

    protected FloatIHistogram2D internalSliceYZ(String title, int indexX1, int indexX2) {
        // Attention: our internal definition of bins has been choosen
        // so that this works properly even if the indeces passed in include
        // the underflow or overflow bins
        if (indexX2 < indexX1)
            throw new IllegalArgumentException("Invalid bin range");

        int yBins = yAxis.bins() + 2;
        int zBins = zAxis.bins() + 2;
        int[][] sliceEntries = new int[yBins][zBins];
        float[][] sliceHeights = new float[yBins][zBins];
        float[][] sliceErrors = new float[yBins][zBins];

        for (int i = indexX1; i <= indexX2; i++) {
            for (int j = 0; j < yBins; j++) {
                for (int k = 0; k < zBins; k++) {
                    sliceEntries[j][k] += entries[i][j][k];
                    sliceHeights[j][k] += heights[i][j][k];
                    sliceErrors[j][k] += errors[i][j][k];
                }
            }
        }
        FloatHistogram2D result = new FloatHistogram2D(title, yAxis, zAxis);
        result.setContents(sliceEntries, sliceHeights, sliceErrors);
        return result;
    }

    public float meanX() {
        return meanX / sumWeight;
    }

    public float meanY() {
        return meanY / sumWeight;
    }

    public float meanZ() {
        return meanZ / sumWeight;
    }

    public void reset() {
        for (int i = 0; i < entries.length; i++)
            for (int j = 0; j < entries[0].length; j++)
                for (int k = 0; j < entries[0][0].length; k++) {
                    entries[i][j][k] = 0;
                    heights[i][j][k] = 0;
                    errors[i][j][k] = 0;
                }
        nEntry = 0;
        sumWeight = 0;
        sumWeightSquared = 0;
        meanX = 0;
        rmsX = 0;
        meanY = 0;
        rmsY = 0;
        meanZ = 0;
        rmsZ = 0;
    }

    public float rmsX() {
        return (float) Math.sqrt(rmsX / sumWeight - meanX * meanX / sumWeight / sumWeight);
    }

    public float rmsY() {
        return (float) Math.sqrt(rmsY / sumWeight - meanY * meanY / sumWeight / sumWeight);
    }

    public float rmsZ() {
        return (float) Math.sqrt(rmsZ / sumWeight - meanZ * meanZ / sumWeight / sumWeight);
    }

    public float sumAllBinHeights() {
        return sumWeight;
    }
}
