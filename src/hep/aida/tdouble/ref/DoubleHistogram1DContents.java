package hep.aida.tdouble.ref;

public class DoubleHistogram1DContents {
    private double[] errors;

    private double[] heights;

    private int[] entries;

    private int nEntry; // total number of times fill called

    private double sumWeight; // Sum of all weights

    private double sumWeightSquared; // Sum of the squares of the weights

    private double mean, rms;

    public DoubleHistogram1DContents(int[] entries, double[] heights, double[] errors, int nEntry, double sumWeight,
            double sumWeightSquared, double mean, double rms) {
        this.entries = entries;
        this.heights = heights;
        this.errors = errors;
        this.nEntry = nEntry;
        this.sumWeight = sumWeight;
        this.sumWeightSquared = sumWeightSquared;
        this.mean = mean;
        this.rms = rms;
    }

    public int[] getEntries() {
        return entries;
    }

    public double[] getHeights() {
        return heights;
    }

    public double[] getErrors() {
        return errors;
    }

    public int getNentry() {
        return nEntry;
    }

    public double getSumWeight() {
        return sumWeight;
    }

    public double getSumWeightSquared() {
        return sumWeightSquared;
    }

    public double getMean() {
        return mean;
    }

    public double getRms() {
        return rms;
    }

}
