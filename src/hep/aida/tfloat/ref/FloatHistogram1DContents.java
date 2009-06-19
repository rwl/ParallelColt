package hep.aida.tfloat.ref;

public class FloatHistogram1DContents {
    private float[] errors;

    private float[] heights;

    private int[] entries;

    private int nEntry; // total number of times fill called

    private float sumWeight; // Sum of all weights

    private float sumWeightSquared; // Sum of the squares of the weights

    private float mean, rms;

    public FloatHistogram1DContents(int[] entries, float[] heights, float[] errors, int nEntry, float sumWeight,
            float sumWeightSquared, float mean, float rms) {
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

    public float[] getHeights() {
        return heights;
    }

    public float[] getErrors() {
        return errors;
    }

    public int getNentry() {
        return nEntry;
    }

    public float getSumWeight() {
        return sumWeight;
    }

    public float getSumWeightSquared() {
        return sumWeightSquared;
    }

    public float getMean() {
        return mean;
    }

    public float getRms() {
        return rms;
    }

}
