package hep.aida.tfloat.ref;

/**
 * Base class for Histogram1D and Histogram2D.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
abstract class FloatHistogram implements hep.aida.tfloat.FloatIHistogram {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    private String title;

    FloatHistogram(String title) {
        this.title = title;
    }

    public String title() {
        return title;
    }
}
