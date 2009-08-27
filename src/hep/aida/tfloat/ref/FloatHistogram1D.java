package hep.aida.tfloat.ref;

import hep.aida.tfloat.FloatIAxis;
import hep.aida.tfloat.FloatIHistogram1D;

import java.util.concurrent.Future;

import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * A reference implementation of hep.aida.IHistogram1D. The goal is to provide a
 * clear implementation rather than the most efficient implementation. However,
 * performance seems fine - filling 1.2 * 10^6 points/sec, both using FixedAxis
 * or VariableAxis.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatHistogram1D extends FloatAbstractHistogram1D implements FloatIHistogram1D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private float[] errors;

    private float[] heights;

    private int[] entries;

    private int nEntry; // total number of times fill called

    private float sumWeight; // Sum of all weights

    private float sumWeightSquared; // Sum of the squares of the weights

    private float mean, rms;

    /**
     * Creates a variable-width histogram. Example:
     * <tt>edges = (0.2, 1.0, 5.0)</tt> yields an axis with 2 in-range bins
     * <tt>[0.2,1.0), [1.0,5.0)</tt> and 2 extra bins
     * <tt>[-inf,0.2), [5.0,inf]</tt>.
     * 
     * @param title
     *            The histogram title.
     * @param edges
     *            the bin boundaries the axis shall have; must be sorted
     *            ascending and must not contain multiple identical elements.
     * @throws IllegalArgumentException
     *             if <tt>edges.length < 1</tt>.
     */
    public FloatHistogram1D(String title, float[] edges) {
        this(title, new FloatVariableAxis(edges));
    }

    /**
     * Creates a histogram with the given axis binning.
     * 
     * @param title
     *            The histogram title.
     * @param axis
     *            The axis description to be used for binning.
     */
    public FloatHistogram1D(String title, FloatIAxis axis) {
        super(title);
        xAxis = axis;
        int bins = axis.bins();
        entries = new int[bins + 2];
        heights = new float[bins + 2];
        errors = new float[bins + 2];
    }

    /**
     * Creates a fixed-width histogram.
     * 
     * @param title
     *            The histogram title.
     * @param bins
     *            The number of bins.
     * @param min
     *            The minimum value on the X axis.
     * @param max
     *            The maximum value on the X axis.
     */
    public FloatHistogram1D(String title, int bins, float min, float max) {
        this(title, new FloatFixedAxis(bins, min, max));
    }

    public int allEntries() // perhaps to be deleted (default
    // impl. in
    // superclass sufficient)
    {
        return nEntry;
    }

    public int binEntries(int index) {
        // return entries[xAxis.map(index)];
        return entries[map(index)];
    }

    public float binError(int index) {
        // return Math.sqrt(errors[xAxis.map(index)]);
        return (float) Math.sqrt(errors[map(index)]);
    }

    public float binHeight(int index) {
        // return heights[xAxis.map(index)];
        return heights[map(index)];
    }

    public float equivalentBinEntries() {
        return sumWeight * sumWeight / sumWeightSquared;
    }

    public void fill(float x) {
        // int bin = xAxis.getBin(x);
        int bin = map(xAxis.coordToIndex(x));
        entries[bin]++;
        heights[bin]++;
        errors[bin]++;
        nEntry++;
        sumWeight++;
        sumWeightSquared++;
        mean += x;
        rms += x * x;
    }

    public void fill(float x, float weight) {
        int bin = map(xAxis.coordToIndex(x));
        entries[bin]++;
        heights[bin] += weight;
        errors[bin] += weight * weight;
        nEntry++;
        sumWeight += weight;
        sumWeightSquared += weight * weight;
        mean += x * weight;
        rms += x * weight * weight;
    }

    public void fill_2D(final float[] data, final int rows, final int columns, final int zero, final int rowStride,
            final int columnStride) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        float[] errors_loc = new float[errors.length];
                        float[] heights_loc = new float[heights.length];
                        int[] entries_loc = new int[entries.length];
                        int nEntry_loc = 0;
                        float sumWeight_loc = 0;
                        float sumWeightSquared_loc = 0;
                        float mean_loc = 0, rms_loc = 0;
                        int idx = zero + firstRow * rowStride;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                int bin = map(xAxis.coordToIndex(data[i]));
                                entries_loc[bin]++;
                                heights_loc[bin]++;
                                errors_loc[bin]++;
                                nEntry_loc++;
                                sumWeight_loc++;
                                sumWeightSquared_loc++;
                                mean_loc += data[i];
                                rms_loc += data[i] * data[i];
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        synchronized (this) {
                            for (int i = 0; i < entries.length; i++) {
                                errors[i] += errors_loc[i];
                                heights[i] += heights_loc[i];
                                entries[i] += entries_loc[i];
                            }
                            nEntry += nEntry_loc;
                            sumWeight += sumWeight_loc;
                            sumWeightSquared += sumWeightSquared_loc;
                            mean += mean_loc;
                            rms += rms_loc;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    int bin = map(xAxis.coordToIndex(data[i]));
                    entries[bin]++;
                    heights[bin]++;
                    errors[bin]++;
                    nEntry++;
                    sumWeight++;
                    sumWeightSquared++;
                    mean += data[i];
                    rms += data[i] * data[i];
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
    }

    public void fill_2D(final float[] data, final float[] weights, final int rows, final int columns, final int zero,
            final int rowStride, final int columnStride) {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (rows * columns >= ConcurrencyUtils.getThreadsBeginN_1D())) {
            nthreads = Math.min(nthreads, rows);
            Future<?>[] futures = new Future[nthreads];
            int k = rows / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstRow = j * k;
                final int lastRow = (j == nthreads - 1) ? rows : firstRow + k;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {

                    public void run() {
                        int idx = zero + firstRow * rowStride;
                        float[] errors_loc = new float[errors.length];
                        float[] heights_loc = new float[heights.length];
                        int[] entries_loc = new int[entries.length];
                        int nEntry_loc = 0;
                        float sumWeight_loc = 0;
                        float sumWeightSquared_loc = 0;
                        float mean_loc = 0, rms_loc = 0;
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int i = idx, c = 0; c < columns; c++) {
                                int bin = map(xAxis.coordToIndex(data[i]));
                                int widx = r * columns + c;
                                float w2 = weights[widx] * weights[widx];
                                entries_loc[bin]++;
                                heights_loc[bin] += weights[widx];
                                errors_loc[bin] += w2;
                                nEntry_loc++;
                                sumWeight_loc += weights[widx];
                                sumWeightSquared_loc += w2;
                                mean_loc += data[i] * weights[widx];
                                rms_loc += data[i] * w2;
                                i += columnStride;
                            }
                            idx += rowStride;
                        }
                        synchronized (this) {
                            for (int i = 0; i < entries.length; i++) {
                                errors[i] += errors_loc[i];
                                heights[i] += heights_loc[i];
                                entries[i] += entries_loc[i];
                            }
                            nEntry += nEntry_loc;
                            sumWeight += sumWeight_loc;
                            sumWeightSquared += sumWeightSquared_loc;
                            mean += mean_loc;
                            rms += rms_loc;
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = zero;
            for (int r = 0; r < rows; r++) {
                for (int i = idx, c = 0; c < columns; c++) {
                    int bin = map(xAxis.coordToIndex(data[i]));
                    int widx = r * columns + c;
                    float w2 = weights[widx] * weights[widx];
                    entries[bin]++;
                    heights[bin] += weights[widx];
                    errors[bin] += w2;
                    nEntry++;
                    sumWeight += weights[widx];
                    sumWeightSquared += w2;
                    mean += data[i] * weights[widx];
                    rms += data[i] * w2;
                    i += columnStride;
                }
                idx += rowStride;
            }
        }
    }

    /**
     * Returns the contents of this histogram.
     * 
     * @return the contents of this histogram
     */
    public FloatHistogram1DContents getContents() {
        return new FloatHistogram1DContents(entries, heights, errors, nEntry, sumWeight, sumWeightSquared, mean, rms);
    }

    public float mean() {
        return mean / sumWeight;
    }

    public void reset() {
        for (int i = 0; i < entries.length; i++) {
            entries[i] = 0;
            heights[i] = 0;
            errors[i] = 0;
        }
        nEntry = 0;
        sumWeight = 0;
        sumWeightSquared = 0;
        mean = 0;
        rms = 0;
    }

    public float rms() {
        return (float) Math.sqrt(rms / sumWeight - mean * mean / sumWeight / sumWeight);
    }

    /**
     * Sets the contents of this histogram
     * 
     * @param contents
     */
    public void setContents(FloatHistogram1DContents contents) {
        this.entries = contents.getEntries();
        this.heights = contents.getHeights();
        this.errors = contents.getErrors();
        this.nEntry = contents.getNentry();
        this.sumWeight = contents.getSumWeight();
        this.sumWeightSquared = contents.getSumWeightSquared();
        this.mean = contents.getMean();
        this.rms = contents.getRms();
    }

    /**
     * Used internally for creating slices and projections
     */
    void setContents(int[] entries, float[] heights, float[] errors) {
        this.entries = entries;
        this.heights = heights;
        this.errors = errors;

        for (int i = 0; i < entries.length; i++) {
            nEntry += entries[i];
            sumWeight += heights[i];
        }
        // TODO: Can we do anything sensible/useful with the other statistics?
        sumWeightSquared = Float.NaN;
        mean = Float.NaN;
        rms = Float.NaN;
    }
}
