package hep.aida.tdouble.ref;

/**
 * Base class for Histogram1D and Histogram2D.
 * 
 * @author Wolfgang Hoschek, Tony Johnson, and others.
 * @version 1.0, 23/03/2000
 */
abstract class DoubleHistogram implements hep.aida.tdouble.DoubleIHistogram {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    private String title;

    DoubleHistogram(String title) {
        this.title = title;
    }

    public String title() {
        return title;
    }
}
