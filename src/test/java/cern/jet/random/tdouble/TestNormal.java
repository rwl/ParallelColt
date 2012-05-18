/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package cern.jet.random.tdouble;

/**
 *
 * @author vvasuki
 */
public class TestNormal {
    public static void main(String[] argv) {
        Normal normal = new Normal(0.0, 1.0, Normal.makeDefaultGenerator());
        double x1 = normal.nextDouble(0, 0);
        double x2 = normal.nextDouble(0, 1);
        System.out.println(x1);
        System.out.println(x2);
    }

}
