/*
    Minpack_f77.java copyright claim:

    This software is based on the public domain MINPACK routines.
    It was translated from FORTRAN to Java by a US government employee 
    on official time.  Thus this software is also in the public domain.


    The translator's mail address is:

    Steve Verrill 
    USDA Forest Products Laboratory
    1 Gifford Pinchot Drive
    Madison, Wisconsin
    53705


    The translator's e-mail address is:

    steve@ws13.fpl.fs.fed.us


***********************************************************************

DISCLAIMER OF WARRANTIES:

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. 
THE TRANSLATOR DOES NOT WARRANT, GUARANTEE OR MAKE ANY REPRESENTATIONS 
REGARDING THE SOFTWARE OR DOCUMENTATION IN TERMS OF THEIR CORRECTNESS, 
RELIABILITY, CURRENCY, OR OTHERWISE. THE ENTIRE RISK AS TO 
THE RESULTS AND PERFORMANCE OF THE SOFTWARE IS ASSUMED BY YOU. 
IN NO CASE WILL ANY PARTY INVOLVED WITH THE CREATION OR DISTRIBUTION 
OF THE SOFTWARE BE LIABLE FOR ANY DAMAGE THAT MAY RESULT FROM THE USE 
OF THIS SOFTWARE.

Sorry about that.

***********************************************************************


History:

Date        Translator        Changes

11/3/00     Steve Verrill     Translated

*/

package optimization;

/**
 * 
 *<p>
 * This class contains Java translations of the MINPACK nonlinear least squares
 * routines. As of November 2000, it does not yet contain the MINPACK solvers of
 * systems of nonlinear equations. They should be added in the Spring of 2001.
 * <p>
 * The original FORTRAN MINPACK package was produced by Burton S. Garbow,
 * Kenneth E. Hillstrom, and Jorge J. More as part of the Argonne National
 * Laboratory MINPACK project, March 1980.
 * 
 *<p>
 * <b>IMPORTANT:</b> The "_f77" suffixes indicate that these routines use
 * FORTRAN style indexing. For example, you will see
 * 
 * <pre>
 *   for (i = 1; i &lt;= n; i++)
 *</pre>
 * 
 * rather than
 * 
 * <pre>
 *   for (i = 0; i &lt; n; i++)
 *</pre>
 * 
 * To use the "_f77" routines you will have to declare your vectors and matrices
 * to be one element larger (e.g., v[101] rather than v[100], and a[101][101]
 * rather than a[100][100]), and you will have to fill elements 1 through n
 * rather than elements 0 through n - 1. Versions of these programs that use
 * C/Java style indexing might eventually be available. They would end with the
 * suffix "_j".
 * 
 *<p>
 * This class was translated by a statistician from FORTRAN versions of lmder
 * and lmdif. It is NOT an official translation. It wastes memory by failing to
 * use the first elements of vectors. When public domain Java optimization
 * routines become available from the people who produced MINPACK, then <b>THE
 * CODE PRODUCED BY THE NUMERICAL ANALYSTS SHOULD BE USED</b>.
 * 
 *<p>
 * Meanwhile, if you have suggestions for improving this code, please contact
 * Steve Verrill at steve@ws13.fpl.fs.fed.us.
 * 
 *@author (translator)Steve Verrill
 *@version .5 --- November 3, 2000
 * 
 */

public class Minpack_f77 extends Object {

    // epsmch is the machine precision

    static final double epsmch = 2.22044604926e-16;

    // minmag is the smallest magnitude

    static final double minmag = 2.22507385852e-308;

    static final double zero = 0.0;
    static final double one = 1.0;
    static final double p0001 = .0001;
    static final double p001 = .001;
    static final double p05 = .05;
    static final double p1 = .1;
    static final double p25 = .25;
    static final double p5 = .5;
    static final double p75 = .75;

    /**
     * 
     *<p>
     * The lmder1_f77 method minimizes the sum of the squares of m nonlinear
     * functions in n variables by a modification of the Levenberg-Marquardt
     * algorithm. This is done by using the more general least-squares solver
     * lmder_f77. The user must provide a method which calculates the functions
     * and the Jacobian.
     *<p>
     * Translated by Steve Verrill on November 17, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param nlls
     *            A class that implements the Lmder_fcn interface (see the
     *            definition in Lmder_fcn.java). See LmderTest_f77.java for an
     *            example of such a class. The class must define a method, fcn,
     *            that must have the form
     * 
     *            public static void fcn(int m, int n, double x[], double
     *            fvec[], double fjac[][], int iflag[])
     * 
     *            If iflag[1] equals 1, fcn calculates the values of the m
     *            functions [residuals] at x and returns this vector in fvec. If
     *            iflag[1] equals 2, fcn calculates the Jacobian at x and
     *            returns this matrix in fjac (and does not alter fvec).
     * 
     *            The value of iflag[1] should not be changed by fcn unless the
     *            user wants to terminate execution of lmder_f77. In this case
     *            set iflag[1] to a negative integer.
     * 
     *@param m
     *            A positive integer set to the number of functions [number of
     *            observations]
     *@param n
     *            A positive integer set to the number of variables [number of
     *            parameters]. n must not exceed m.
     *@param x
     *            On input, it contains the initial estimate of the solution
     *            vector [the least squares parameters]. On output it contains
     *            the final estimate of the solution vector.
     *@param fvec
     *            An output vector that contains the m functions [residuals]
     *            evaluated at x.
     *@param fjac
     *            An output m by n array. The upper n by n submatrix of fjac
     *            contains an upper triangular matrix R with diagonal elements
     *            of nonincreasing magnitude such that
     * 
     *            <pre>
     *                 t    t         t
     *                P (jac jac)P = R R,
     *</pre>
     * 
     *            where P is a permutation matrix and jac is the final
     *            calculated Jacobian. Column j of P is column ipvt[j] of the
     *            identity matrix. The lower trapezoidal part of fjac contains
     *            information generated during the computation of R.
     *@param tol
     *            tol is a nonnegative input variable. Termination occurs when
     *            the algorithm estimates either that the relative error in the
     *            sum of squares is at most tol or that the relative error
     *            between x and the solution is at most tol.
     *@param info
     *            An integer output variable. If the user has terminated
     *            execution, info is set to the (negative) value of iflag[1].
     *            See description of fcn. Otherwise, info is set as follows.
     * 
     *            info = 0 improper input parameters.
     * 
     *            info = 1 algorithm estimates that the relative error in the
     *            sum of squares is at most tol.
     * 
     *            info = 2 algorithm estimates that the relative error between x
     *            and the solution is at most tol.
     * 
     *            info = 3 conditions for info = 1 and info = 2 both hold.
     * 
     *            info = 4 fvec is orthogonal to the columns of the Jacobian to
     *            machine precision.
     * 
     *            info = 5 number of calls to fcn with iflag[1] = 1 has reached
     *            100*(n+1).
     * 
     *            info = 6 tol is too small. No further reduction in the sum of
     *            squares is possible.
     * 
     *            info = 7 tol is too small. No further improvement in the
     *            approximate solution x is possible.
     *@param ipvt
     *            An integer output array of length n. ipvt defines a
     *            permutation matrix P such that jac*P = QR, where jac is the
     *            final calculated Jacobian, Q is orthogonal (not stored), and R
     *            is upper triangular with diagonal elements of nonincreasing
     *            magnitude. Column j of P is column ipvt[j] of the identity
     *            matrix.
     * 
     */

    public static void lmder1_f77(Lmder_fcn nlls, int m, int n, double x[], double fvec[], double fjac[][], double tol,
            int info[], int ipvt[]) {

        /*

        Here is a copy of the lmder1 FORTRAN documentation:


              subroutine lmder1(fcn,m,n,x,fvec,fjac,ldfjac,tol,info,ipvt,wa,
             *                  lwa)

              integer m,n,ldfjac,info,lwa
              integer ipvt(n)
              double precision tol
              double precision x(n),fvec(m),fjac(ldfjac,n),wa(lwa)
              external fcn

        c     **********
        c
        c     subroutine lmder1
        c
        c     the purpose of lmder1 is to minimize the sum of the squares of
        c     m nonlinear functions in n variables by a modification of the
        c     levenberg-marquardt algorithm. this is done by using the more
        c     general least-squares solver lmder. the user must provide a
        c     subroutine which calculates the functions and the jacobian.
        c
        c     the subroutine statement is
        c
        c       subroutine lmder1(fcn,m,n,x,fvec,fjac,ldfjac,tol,info,
        c                         ipvt,wa,lwa)
        c
        c     where
        c
        c       fcn is the name of the user-supplied subroutine which
        c         calculates the functions and the jacobian. fcn must
        c         be declared in an external statement in the user
        c         calling program, and should be written as follows.
        c
        c         subroutine fcn(m,n,x,fvec,fjac,ldfjac,iflag)
        c         integer m,n,ldfjac,iflag
        c         double precision x(n),fvec(m),fjac(ldfjac,n)
        c         ----------
        c         if iflag = 1 calculate the functions at x and
        c         return this vector in fvec. do not alter fjac.
        c         if iflag = 2 calculate the jacobian at x and
        c         return this matrix in fjac. do not alter fvec.
        c         ----------
        c         return
        c         end
        c
        c         the value of iflag should not be changed by fcn unless
        c         the user wants to terminate execution of lmder1.
        c         in this case set iflag to a negative integer.
        c
        c       m is a positive integer input variable set to the number
        c         of functions.
        c
        c       n is a positive integer input variable set to the number
        c         of variables. n must not exceed m.
        c
        c       x is an array of length n. on input x must contain
        c         an initial estimate of the solution vector. on output x
        c         contains the final estimate of the solution vector.
        c
        c       fvec is an output array of length m which contains
        c         the functions evaluated at the output x.
        c
        c       fjac is an output m by n array. the upper n by n submatrix
        c         of fjac contains an upper triangular matrix r with
        c         diagonal elements of nonincreasing magnitude such that
        c
        c                t     t           t
        c               p *(jac *jac)*p = r *r,
        c
        c         where p is a permutation matrix and jac is the final
        c         calculated jacobian. column j of p is column ipvt(j)
        c         (see below) of the identity matrix. the lower trapezoidal
        c         part of fjac contains information generated during
        c         the computation of r.
        c
        c       ldfjac is a positive integer input variable not less than m
        c         which specifies the leading dimension of the array fjac.
        c
        c       tol is a nonnegative input variable. termination occurs
        c         when the algorithm estimates either that the relative
        c         error in the sum of squares is at most tol or that
        c         the relative error between x and the solution is at
        c         most tol.
        c
        c       info is an integer output variable. if the user has
        c         terminated execution, info is set to the (negative)
        c         value of iflag. see description of fcn. otherwise,
        c         info is set as follows.
        c
        c         info = 0  improper input parameters.
        c
        c         info = 1  algorithm estimates that the relative error
        c                   in the sum of squares is at most tol.
        c
        c         info = 2  algorithm estimates that the relative error
        c                   between x and the solution is at most tol.
        c
        c         info = 3  conditions for info = 1 and info = 2 both hold.
        c
        c         info = 4  fvec is orthogonal to the columns of the
        c                   jacobian to machine precision.
        c
        c         info = 5  number of calls to fcn with iflag = 1 has
        c                   reached 100*(n+1).
        c
        c         info = 6  tol is too small. no further reduction in
        c                   the sum of squares is possible.
        c
        c         info = 7  tol is too small. no further improvement in
        c                   the approximate solution x is possible.
        c
        c       ipvt is an integer output array of length n. ipvt
        c         defines a permutation matrix p such that jac*p = q*r,
        c         where jac is the final calculated jacobian, q is
        c         orthogonal (not stored), and r is upper triangular
        c         with diagonal elements of nonincreasing magnitude.
        c         column j of p is column ipvt(j) of the identity matrix.
        c
        c       wa is a work array of length lwa.
        c
        c       lwa is a positive integer input variable not less than 5*n+m.
        c
        c     subprograms called
        c
        c       user-supplied ...... fcn
        c
        c       minpack-supplied ... lmder
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int maxfev, mode, nprint;

        int nfev[] = new int[2];
        int njev[] = new int[2];

        double diag[] = new double[n + 1];
        double qtf[] = new double[n + 1];

        //      double factor,ftol,gtol,xtol,zero;
        double factor, ftol, gtol, xtol;

        factor = 1.0e+2;
        //      zero = 0.0;

        info[1] = 0;

        // Check the input parameters for errors.

        if (n <= 0 || m < n || tol < zero) {

            return;

        } else {

            // Call lmder_f77.

            maxfev = 100 * (n + 1);
            ftol = tol;
            xtol = tol;
            gtol = zero;
            mode = 1;
            nprint = 0;

            Minpack_f77.lmder_f77(nlls, m, n, x, fvec, fjac, ftol, xtol, gtol, maxfev, diag, mode, factor, nprint,
                    info, nfev, njev, ipvt, qtf);

            if (info[1] == 8)
                info[1] = 4;

            return;

        }

    }

    /**
     * 
     *<p>
     * The lmder_f77 method minimizes the sum of the squares of m nonlinear
     * functions in n variables by a modification of the Levenberg-Marquardt
     * algorithm. The user must provide a method which calculates the functions
     * and the Jacobian.
     *<p>
     * Translated by Steve Verrill on November 3, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param nlls
     *            A class that implements the Lmder_fcn interface (see the
     *            definition in Lmder_fcn.java). See LmderTest_f77.java for an
     *            example of such a class. The class must define a method, fcn,
     *            that must have the form
     * 
     *            public static void fcn(int m, int n, double x[], double
     *            fvec[], double fjac[][], int iflag[])
     * 
     *            If iflag[1] equals 1, fcn calculates the values of the m
     *            functions [residuals] at x and returns this vector in fvec. If
     *            iflag[1] equals 2, fcn calculates the Jacobian at x and
     *            returns this matrix in fjac (and does not alter fvec).
     * 
     *            The value of iflag[1] should not be changed by fcn unless the
     *            user wants to terminate execution of lmder_f77. In this case
     *            set iflag[1] to a negative integer.
     * 
     *@param m
     *            A positive integer set to the number of functions [number of
     *            observations]
     *@param n
     *            A positive integer set to the number of variables [number of
     *            parameters]. n must not exceed m.
     *@param x
     *            On input, it contains the initial estimate of the solution
     *            vector [the least squares parameters]. On output it contains
     *            the final estimate of the solution vector.
     *@param fvec
     *            An output vector that contains the m functions [residuals]
     *            evaluated at x.
     *@param fjac
     *            An output m by n array. The upper n by n submatrix of fjac
     *            contains an upper triangular matrix R with diagonal elements
     *            of nonincreasing magnitude such that
     * 
     *            <pre>
     *                 t    t         t
     *                P (jac jac)P = R R,
     *</pre>
     * 
     *            where P is a permutation matrix and jac is the final
     *            calculated Jacobian. Column j of P is column ipvt[j] of the
     *            identity matrix. The lower trapezoidal part of fjac contains
     *            information generated during the computation of R.
     *@param ftol
     *            A nonnegative input variable. Termination occurs when both the
     *            actual and predicted relative reductions in the sum of squares
     *            are at most ftol. Therefore, ftol measures the relative error
     *            desired in the sum of squares.
     *@param xtol
     *            A nonnegative input variable. Termination occurs when the
     *            relative error between two consecutive iterates is at most
     *            xtol. Therefore, xtol measures the relative error desired in
     *            the approximate solution.
     *@param gtol
     *            A nonnegative input variable. Termination occurs when the
     *            cosine of the angle between fvec and any column of the
     *            Jacobian is at most gtol in absolute value. Therefore, gtol
     *            measures the orthogonality desired between the function vector
     *            and the columns of the Jacobian.
     *@param maxfev
     *            A positive integer input variable. Termination occurs when the
     *            number of calls to fcn with iflag[1] = 1 has reached maxfev.
     *@param diag
     *            An vector of length n. If mode = 1 (see below), diag is
     *            internally set. If mode = 2, diag must contain positive
     *            entries that serve as multiplicative scale factors for the
     *            variables.
     *@param mode
     *            If mode = 1, the variables will be scaled internally. If mode
     *            = 2, the scaling is specified by the input diag. Other values
     *            of mode are equivalent to mode = 1.
     *@param factor
     *            A positive input variable used in determining the initial step
     *            bound. This bound is set to the product of factor and the
     *            euclidean norm of diag*x if nonzero, or else to factor itself.
     *            In most cases factor should lie in the interval (.1,100). 100
     *            is a generally recommended value.
     *@param nprint
     *            An integer input variable that enables controlled printing of
     *            iterates if it is positive. In this case, fcn is called with
     *            iflag[1] = 0 at the beginning of the first iteration and every
     *            nprint iterations thereafter and immediately prior to return,
     *            with x, fvec, and fjac available for printing. fvec and fjac
     *            should not be altered. If nprint is not positive, no special
     *            calls of fcn with iflag[1] = 0 are made.
     *@param info
     *            An integer output variable. If the user has terminated
     *            execution, info is set to the (negative) value of iflag[1].
     *            See description of fcn. Otherwise, info is set as follows.
     * 
     *            info = 0 improper input parameters.
     * 
     *            info = 1 both actual and predicted relative reductions in the
     *            sum of squares are at most ftol.
     * 
     *            info = 2 relative error between two consecutive iterates is at
     *            most xtol.
     * 
     *            info = 3 conditions for info = 1 and info = 2 both hold.
     * 
     *            info = 4 the cosine of the angle between fvec and any column
     *            of the Jacobian is at most gtol in absolute value.
     * 
     *            info = 5 number of calls to fcn with iflag[1] = 1 has reached
     *            maxfev.
     * 
     *            info = 6 ftol is too small. no further reduction in the sum of
     *            squares is possible.
     * 
     *            info = 7 xtol is too small. no further improvement in the
     *            approximate solution x is possible.
     * 
     *            info = 8 gtol is too small. fvec is orthogonal to the columns
     *            of the Jacobian to machine precision.
     * 
     *@param nfev
     *            An integer output variable set to the number of calls to fcn
     *            with iflag[1] = 1.
     *@param njev
     *            An integer output variable set to the number of calls to fcn
     *            with iflag[1] = 2.
     *@param ipvt
     *            An integer output array of length n. ipvt defines a
     *            permutation matrix P such that jac*P = QR, where jac is the
     *            final calculated Jacobian, Q is orthogonal (not stored), and R
     *            is upper triangular with diagonal elements of nonincreasing
     *            magnitude. column j of P is column ipvt[j] of the identity
     *            matrix.
     * 
     *@param qtf
     *            An output array of length n which contains the first n
     *            elements of the vector (Q transpose)fvec.
     * 
     * 
     */

    /*
    Note that since Java passes primitive types by value rather than
    by reference, if they need
    to return a value, they need to be passed as arrays (here we
    place the actual value in location [1]).  For example, info
    is passed as info[].
    */

    public static void lmder_f77(Lmder_fcn nlls, int m, int n, double x[], double fvec[], double fjac[][], double ftol,
            double xtol, double gtol, int maxfev, double diag[], int mode, double factor, int nprint, int info[],
            int nfev[], int njev[], int ipvt[], double qtf[]) {

        /*

        Here is a copy of the lmder FORTRAN documentation:


              subroutine lmder(fcn,m,n,x,fvec,fjac,ldfjac,ftol,xtol,gtol,
             *                 maxfev,diag,mode,factor,nprint,info,nfev,njev,
             *                 ipvt,qtf,wa1,wa2,wa3,wa4)
              integer m,n,ldfjac,maxfev,mode,nprint,info,nfev,njev
              integer ipvt(n)
              double precision ftol,xtol,gtol,factor
              double precision x(n),fvec(m),fjac(ldfjac,n),diag(n),qtf(n),
             *                 wa1(n),wa2(n),wa3(n),wa4(m)
        c
        c     subroutine lmder
        c
        c     the purpose of lmder is to minimize the sum of the squares of
        c     m nonlinear functions in n variables by a modification of
        c     the Levenberg-Marquardt algorithm. the user must provide a
        c     subroutine which calculates the functions and the Jacobian.
        c
        c     the subroutine statement is
        c
        c       subroutine lmder(fcn,m,n,x,fvec,fjac,ldfjac,ftol,xtol,gtol,
        c                        maxfev,diag,mode,factor,nprint,info,nfev,
        c                        njev,ipvt,qtf,wa1,wa2,wa3,wa4)
        c
        c     where
        c
        c       fcn is the name of the user-supplied subroutine which
        c         calculates the functions and the Jacobian. fcn must
        c         be declared in an external statement in the user
        c         calling program, and should be written as follows.
        c
        c         subroutine fcn(m,n,x,fvec,fjac,ldfjac,iflag)
        c         integer m,n,ldfjac,iflag
        c         double precision x(n),fvec(m),fjac(ldfjac,n)
        c         ----------
        c         if iflag = 1 calculate the functions at x and
        c         return this vector in fvec. do not alter fjac.
        c         if iflag = 2 calculate the Jacobian at x and
        c         return this matrix in fjac. do not alter fvec.
        c         ----------
        c         return
        c         end
        c
        c         the value of iflag should not be changed by fcn unless
        c         the user wants to terminate execution of lmder.
        c         in this case set iflag to a negative integer.
        c
        c       m is a positive integer input variable set to the number
        c         of functions.
        c
        c       n is a positive integer input variable set to the number
        c         of variables. n must not exceed m.
        c
        c       x is an array of length n. on input x must contain
        c         an initial estimate of the solution vector. on output x
        c         contains the final estimate of the solution vector.
        c
        c       fvec is an output array of length m which contains
        c         the functions evaluated at the output x.
        c
        c       fjac is an output m by n array. the upper n by n submatrix
        c         of fjac contains an upper triangular matrix r with
        c         diagonal elements of nonincreasing magnitude such that
        c
        c                t     t           t
        c               p *(jac *jac)*p = r *r,
        c
        c         where p is a permutation matrix and jac is the final
        c         calculated Jacobian. column j of p is column ipvt(j)
        c         (see below) of the identity matrix. the lower trapezoidal
        c         part of fjac contains information generated during
        c         the computation of r.
        c
        c       ldfjac is a positive integer input variable not less than m
        c         which specifies the leading dimension of the array fjac.
        c
        c       ftol is a nonnegative input variable. termination
        c         occurs when both the actual and predicted relative
        c         reductions in the sum of squares are at most ftol.
        c         therefore, ftol measures the relative error desired
        c         in the sum of squares.
        c
        c       xtol is a nonnegative input variable. termination
        c         occurs when the relative error between two consecutive
        c         iterates is at most xtol. therefore, xtol measures the
        c         relative error desired in the approximate solution.
        c
        c       gtol is a nonnegative input variable. termination
        c         occurs when the cosine of the angle between fvec and
        c         any column of the Jacobian is at most gtol in absolute
        c         value. therefore, gtol measures the orthogonality
        c         desired between the function vector and the columns
        c         of the Jacobian.
        c
        c       maxfev is a positive integer input variable. termination
        c         occurs when the number of calls to fcn with iflag = 1
        c         has reached maxfev.
        c
        c       diag is an array of length n. if mode = 1 (see
        c         below), diag is internally set. if mode = 2, diag
        c         must contain positive entries that serve as
        c         multiplicative scale factors for the variables.
        c
        c       mode is an integer input variable. if mode = 1, the
        c         variables will be scaled internally. if mode = 2,
        c         the scaling is specified by the input diag. other
        c         values of mode are equivalent to mode = 1.
        c
        c       factor is a positive input variable used in determining the
        c         initial step bound. this bound is set to the product of
        c         factor and the euclidean norm of diag*x if nonzero, or else
        c         to factor itself. in most cases factor should lie in the
        c         interval (.1,100.).100. is a generally recommended value.
        c
        c       nprint is an integer input variable that enables controlled
        c         printing of iterates if it is positive. in this case,
        c         fcn is called with iflag = 0 at the beginning of the first
        c         iteration and every nprint iterations thereafter and
        c         immediately prior to return, with x, fvec, and fjac
        c         available for printing. fvec and fjac should not be
        c         altered. if nprint is not positive, no special calls
        c         of fcn with iflag = 0 are made.
        c
        c       info is an integer output variable. if the user has
        c         terminated execution, info is set to the (negative)
        c         value of iflag. see description of fcn. otherwise,
        c         info is set as follows.
        c
        c         info = 0  improper input parameters.
        c
        c         info = 1  both actual and predicted relative reductions
        c                   in the sum of squares are at most ftol.
        c
        c         info = 2  relative error between two consecutive iterates
        c                   is at most xtol.
        c
        c         info = 3  conditions for info = 1 and info = 2 both hold.
        c
        c         info = 4  the cosine of the angle between fvec and any
        c                   column of the Jacobian is at most gtol in
        c                   absolute value.
        c
        c         info = 5  number of calls to fcn with iflag = 1 has
        c                   reached maxfev.
        c
        c         info = 6  ftol is too small. no further reduction in
        c                   the sum of squares is possible.
        c
        c         info = 7  xtol is too small. no further improvement in
        c                   the approximate solution x is possible.
        c
        c         info = 8  gtol is too small. fvec is orthogonal to the
        c                   columns of the Jacobian to machine precision.
        c
        c       nfev is an integer output variable set to the number of
        c         calls to fcn with iflag = 1.
        c
        c       njev is an integer output variable set to the number of
        c         calls to fcn with iflag = 2.
        c
        c       ipvt is an integer output array of length n. ipvt
        c         defines a permutation matrix p such that jac*p = q*r,
        c         where jac is the final calculated Jacobian, q is
        c         orthogonal (not stored), and r is upper triangular
        c         with diagonal elements of nonincreasing magnitude.
        c         column j of p is column ipvt(j) of the identity matrix.
        c
        c       qtf is an output array of length n which contains
        c         the first n elements of the vector (q transpose)*fvec.
        c
        c       wa1, wa2, and wa3 are work arrays of length n.
        c
        c       wa4 is a work array of length m.
        c
        c     subprograms called
        c
        c       user-supplied ...... fcn
        c
        c       minpack-supplied ... dpmpar,enorm,lmpar,qrfac
        c
        c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt,mod
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c

        */

        int i, iter, j, l;
        //      double actred,delta,dirder,fnorm,fnorm1,gnorm,
        //             one,pnorm,prered,p1,p5,p25,p75,p0001,ratio,
        //             sum,temp,temp1,temp2,xnorm,zero;

        double actred, delta, dirder, fnorm, fnorm1, gnorm, pnorm, prered, ratio, sum, temp, temp1, temp2, xnorm;

        double par[] = new double[2];

        boolean doneout, donein;

        int iflag[] = new int[2];
        double wa1[] = new double[n + 1];
        double wa2[] = new double[n + 1];
        double wa3[] = new double[n + 1];
        double wa4[] = new double[m + 1];

        // Java compiler complains if delta and xnorm are not initialized

        delta = 0.0;
        xnorm = 0.0;

        //      one = 1.0;
        //      p1 = .1;
        //      p5 = .5;
        //      p25 = .25;
        //      p75 = .75;
        //      p0001 = .0001;
        //      zero = 0.0;

        info[1] = 0;
        iflag[1] = 0;
        nfev[1] = 0;
        njev[1] = 0;

        // Check the input parameters for errors.

        if (n <= 0 || m < n || ftol < zero || xtol < zero || gtol < zero || maxfev <= 0 || factor <= zero) {

            // Termination 

            if (nprint > 0) {

                nlls.fcn(m, n, x, fvec, fjac, iflag);

            }

            return;

        }

        if (mode == 2) {

            for (j = 1; j <= n; j++) {

                if (diag[j] <= zero) {

                    // Termination

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, fjac, iflag);

                    }

                    return;

                }

            }

        }

        // Evaluate the function at the starting point
        // and calculate its norm.

        iflag[1] = 1;

        nlls.fcn(m, n, x, fvec, fjac, iflag);

        nfev[1] = 1;

        if (iflag[1] < 0) {

            // Termination

            info[1] = iflag[1];
            iflag[1] = 0;

            if (nprint > 0) {

                nlls.fcn(m, n, x, fvec, fjac, iflag);

            }

            return;

        }

        fnorm = Minpack_f77.enorm_f77(m, fvec);

        // Initialize Levenberg-Marquardt parameter and iteration counter.

        par[1] = zero;
        iter = 1;

        // Beginning of the outer loop.

        doneout = false;

        while (!doneout) {

            // 30 continue

            // Calculate the Jacobian matrix.

            iflag[1] = 2;

            nlls.fcn(m, n, x, fvec, fjac, iflag);

            njev[1]++;

            if (iflag[1] < 0) {

                // Termination

                info[1] = iflag[1];
                iflag[1] = 0;

                if (nprint > 0) {

                    nlls.fcn(m, n, x, fvec, fjac, iflag);

                }

                return;

            }

            // If requested, call fcn to enable printing of iterates.

            if (nprint > 0) {

                iflag[1] = 0;

                if ((iter - 1) % nprint == 0) {

                    nlls.fcn(m, n, x, fvec, fjac, iflag);

                }

                if (iflag[1] < 0) {

                    // Termination

                    info[1] = iflag[1];
                    iflag[1] = 0;

                    nlls.fcn(m, n, x, fvec, fjac, iflag);

                    return;

                }

            }

            // Compute the qr factorization of the Jacobian.

            Minpack_f77.qrfac_f77(m, n, fjac, true, ipvt, wa1, wa2, wa3);

            // On the first iteration and if mode is 1, scale according
            // to the norms of the columns of the initial Jacobian.

            if (iter == 1) {

                if (mode != 2) {

                    for (j = 1; j <= n; j++) {

                        diag[j] = wa2[j];

                        if (wa2[j] == zero)
                            diag[j] = one;

                    }

                }

                // On the first iteration, calculate the norm of the scaled x
                // and initialize the step bound delta.

                for (j = 1; j <= n; j++) {

                    wa3[j] = diag[j] * x[j];

                }

                xnorm = Minpack_f77.enorm_f77(n, wa3);

                delta = factor * xnorm;

                if (delta == zero)
                    delta = factor;

            }

            // Form (q transpose)*fvec and store the first n components in
            // qtf.

            for (i = 1; i <= m; i++)
                wa4[i] = fvec[i];

            for (j = 1; j <= n; j++) {

                if (fjac[j][j] != zero) {

                    sum = zero;

                    for (i = j; i <= m; i++)
                        sum += fjac[i][j] * wa4[i];

                    temp = -sum / fjac[j][j];

                    for (i = j; i <= m; i++)
                        wa4[i] += fjac[i][j] * temp;

                }

                fjac[j][j] = wa1[j];
                qtf[j] = wa4[j];

            }

            // Compute the norm of the scaled gradient.

            gnorm = zero;

            if (fnorm != zero) {

                for (j = 1; j <= n; j++) {

                    l = ipvt[j];

                    if (wa2[l] != zero) {

                        sum = zero;

                        for (i = 1; i <= j; i++)
                            sum += fjac[i][j] * (qtf[i] / fnorm);

                        gnorm = Math.max(gnorm, Math.abs(sum / wa2[l]));

                    }

                }

            }

            // Test for convergence of the gradient norm.

            if (gnorm <= gtol)
                info[1] = 4;

            if (info[1] != 0) {

                // Termination

                if (iflag[1] < 0)
                    info[1] = iflag[1];
                iflag[1] = 0;

                if (nprint > 0) {

                    nlls.fcn(m, n, x, fvec, fjac, iflag);

                }

                return;

            }

            // Rescale if necessary.

            if (mode != 2) {

                for (j = 1; j <= n; j++) {

                    diag[j] = Math.max(diag[j], wa2[j]);

                }

            }

            // Beginning of the inner loop.

            donein = false;

            while (!donein) {

                // 200    continue

                // Determine the Levenberg-Marquardt parameter.

                Minpack_f77.lmpar_f77(n, fjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4);

                // Store the direction p and x + p. calculate the norm of p.

                for (j = 1; j <= n; j++) {

                    wa1[j] = -wa1[j];
                    wa2[j] = x[j] + wa1[j];
                    wa3[j] = diag[j] * wa1[j];

                }

                pnorm = Minpack_f77.enorm_f77(n, wa3);

                // On the first iteration, adjust the initial step bound.

                if (iter == 1)
                    delta = Math.min(delta, pnorm);

                // Evaluate the function at x + p and calculate its norm.

                iflag[1] = 1;

                nlls.fcn(m, n, wa2, wa4, fjac, iflag);

                nfev[1]++;

                if (iflag[1] < 0) {

                    // Termination

                    info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, fjac, iflag);

                    }

                    return;

                }

                fnorm1 = Minpack_f77.enorm_f77(m, wa4);

                // Compute the scaled actual reduction.

                actred = -one;

                if (p1 * fnorm1 < fnorm)
                    actred = one - (fnorm1 / fnorm) * (fnorm1 / fnorm);

                // Compute the scaled predicted reduction and
                // the scaled directional derivative.

                for (j = 1; j <= n; j++) {

                    wa3[j] = zero;
                    l = ipvt[j];
                    temp = wa1[l];

                    for (i = 1; i <= j; i++)
                        wa3[i] += fjac[i][j] * temp;

                }

                temp1 = Minpack_f77.enorm_f77(n, wa3) / fnorm;
                temp2 = (Math.sqrt(par[1]) * pnorm) / fnorm;

                prered = temp1 * temp1 + temp2 * temp2 / p5;
                dirder = -(temp1 * temp1 + temp2 * temp2);

                // Compute the ratio of the actual to the predicted
                // reduction.

                ratio = zero;
                if (prered != zero)
                    ratio = actred / prered;

                // Update the step bound.

                if (ratio <= p25) {

                    if (actred >= zero) {

                        temp = p5;

                    } else {

                        temp = p5 * dirder / (dirder + p5 * actred);

                    }

                    if (p1 * fnorm1 >= fnorm || temp < p1)
                        temp = p1;

                    delta = temp * Math.min(delta, pnorm / p1);

                    par[1] /= temp;

                } else {

                    if (par[1] == zero || ratio >= p75) {

                        delta = pnorm / p5;
                        par[1] *= p5;

                    }

                }

                // Test for successful iteration.

                if (ratio >= p0001) {

                    // Successful iteration.  Update x, fvec, and their norms.

                    for (j = 1; j <= n; j++) {

                        x[j] = wa2[j];
                        wa2[j] = diag[j] * x[j];

                    }

                    for (i = 1; i <= m; i++)
                        fvec[i] = wa4[i];

                    xnorm = Minpack_f77.enorm_f77(n, wa2);

                    fnorm = fnorm1;

                    iter++;

                }

                // Tests for convergence.

                if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one)
                    info[1] = 1;

                if (delta <= xtol * xnorm)
                    info[1] = 2;

                if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one && info[1] == 2)
                    info[1] = 3;

                if (info[1] != 0) {

                    // Termination

                    if (iflag[1] < 0)
                        info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, fjac, iflag);

                    }

                    return;

                }

                // Tests for termination and stringent tolerances.

                if (nfev[1] >= maxfev)
                    info[1] = 5;

                if (Math.abs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= one)
                    info[1] = 6;

                if (delta <= epsmch * xnorm)
                    info[1] = 7;

                if (gnorm <= epsmch)
                    info[1] = 8;

                if (info[1] != 0) {

                    // Termination

                    if (iflag[1] < 0)
                        info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, fjac, iflag);

                    }

                    return;

                }

                // End of the inner loop.  Repeat if iteration unsuccessful.

                if (ratio >= p0001)
                    donein = true;

            }

            // End of the outer loop.

        }

    }

    /**
     * 
     *<p>
     *The enorm_f77 method calculates the Euclidean norm of a vector.
     *<p>
     * Translated by Steve Verrill on November 14, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param n
     *            The length of the vector, x.
     *@param x
     *            The vector whose Euclidean norm is to be calculated.
     * 
     */

    public static double enorm_f77(int n, double x[]) {

        /*

        Here is a copy of the enorm FORTRAN documentation:


              double precision function enorm(n,x)
              integer n
              double precision x(n)

        c     **********
        c
        c     function enorm
        c
        c     given an n-vector x, this function calculates the
        c     euclidean norm of x.
        c
        c     the euclidean norm is computed by accumulating the sum of
        c     squares in three different sums. the sums of squares for the
        c     small and large components are scaled so that no overflows
        c     occur. non-destructive underflows are permitted. underflows
        c     and overflows do not occur in the computation of the unscaled
        c     sum of squares for the intermediate components.
        c     the definitions of small, intermediate and large components
        c     depend on two constants, rdwarf and rgiant. the main
        c     restrictions on these constants are that rdwarf**2 not
        c     underflow and rgiant**2 not overflow. the constants
        c     given here are suitable for every known computer.
        c
        c     the function statement is
        c
        c       double precision function enorm(n,x)
        c
        c     where
        c
        c       n is a positive integer input variable.
        c
        c       x is an input array of length n.
        c
        c     subprograms called
        c
        c       fortran-supplied ... dabs,dsqrt
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i;
        //      double agiant,floatn,one,rdwarf,rgiant,s1,s2,s3,xabs,
        //            x1max,x3max,zero;
        double agiant, floatn, rdwarf, rgiant, s1, s2, s3, xabs, x1max, x3max;
        double enorm;

        //      one = 1.0;
        //      zero = 0.0;
        rdwarf = 3.834e-20;
        rgiant = 1.304e+19;

        s1 = zero;
        s2 = zero;
        s3 = zero;
        x1max = zero;
        x3max = zero;
        floatn = n;
        agiant = rgiant / floatn;

        for (i = 1; i <= n; i++) {

            xabs = Math.abs(x[i]);

            if (xabs <= rdwarf || xabs >= agiant) {

                if (xabs > rdwarf) {

                    // Sum for large components.

                    if (xabs > x1max) {

                        s1 = one + s1 * (x1max / xabs) * (x1max / xabs);
                        x1max = xabs;

                    } else {

                        s1 += (xabs / x1max) * (xabs / x1max);

                    }

                } else {

                    // Sum for small components.

                    if (xabs > x3max) {

                        s3 = one + s3 * (x3max / xabs) * (x3max / xabs);
                        x3max = xabs;

                    } else {

                        if (xabs != zero)
                            s3 += (xabs / x3max) * (xabs / x3max);

                    }

                }

            } else {

                // Sum for intermediate components.

                s2 += xabs * xabs;

            }

        }

        // Calculation of norm.

        if (s1 != zero) {

            enorm = x1max * Math.sqrt(s1 + (s2 / x1max) / x1max);

        } else {

            if (s2 != zero) {

                if (s2 >= x3max) {

                    enorm = Math.sqrt(s2 * (one + (x3max / s2) * (x3max * s3)));

                } else {

                    enorm = Math.sqrt(x3max * ((s2 / x3max) + (x3max * s3)));

                }

            } else {

                enorm = x3max * Math.sqrt(s3);

            }

        }

        return enorm;

    }

    /**
     * 
     *<p>
     * The qrfac_f77 method uses Householder transformations with column
     * pivoting (optional) to compute a QR factorization of the m by n matrix A.
     * That is, qrfac_f77 determines an orthogonal matrix Q, a permutation
     * matrix P, and an upper trapezoidal matrix R with diagonal elements of
     * nonincreasing magnitude, such that AP = QR.
     *<p>
     * Translated by Steve Verrill on November 17, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param m
     *            The number of rows of A.
     *@param n
     *            The number of columns of A.
     *@param a
     *            A is an m by n array. On input A contains the matrix for which
     *            the QR factorization is to be computed. On output the strict
     *            upper trapezoidal part of A contains the strict upper
     *            trapezoidal part of R, and the lower trapezoidal part of A
     *            contains a factored form of Q.
     *@param pivot
     *            pivot is a logical input variable. If pivot is set true, then
     *            column pivoting is enforced. If pivot is set false, then no
     *            column pivoting is done.
     *@param ipvt
     *            ipvt is an integer output array. ipvt defines the permutation
     *            matrix P such that A*P = Q*R. Column j of P is column ipvt[j]
     *            of the identity matrix. If pivot is false, ipvt is not
     *            referenced.
     *@param rdiag
     *            rdiag is an output array of length n which contains the
     *            diagonal elements of R.
     *@param acnorm
     *            acnorm is an output array of length n which contains the norms
     *            of the corresponding columns of the input matrix A.
     *@param wa
     *            wa is a work array of length n.
     * 
     * 
     */

    public static void qrfac_f77(int m, int n, double a[][], boolean pivot, int ipvt[], double rdiag[],
            double acnorm[], double wa[]) {

        /*

        Here is a copy of the qrfac FORTRAN documentation:


              subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)

              integer m,n,lda,lipvt
              integer ipvt(lipvt)
              logical pivot
              double precision a(lda,n),rdiag(n),acnorm(n),wa(n)

        c     **********
        c
        c     subroutine qrfac
        c
        c     this subroutine uses householder transformations with column
        c     pivoting (optional) to compute a qr factorization of the
        c     m by n matrix a. that is, qrfac determines an orthogonal
        c     matrix q, a permutation matrix p, and an upper trapezoidal
        c     matrix r with diagonal elements of nonincreasing magnitude,
        c     such that a*p = q*r. the householder transformation for
        c     column k, k = 1,2,...,min(m,n), is of the form
        c
        c                           t
        c           i - (1/u(k))*u*u
        c
        c     where u has zeros in the first k-1 positions. the form of
        c     this transformation and the method of pivoting first
        c     appeared in the corresponding linpack subroutine.
        c
        c     the subroutine statement is
        c
        c       subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)
        c
        c     where
        c
        c       m is a positive integer input variable set to the number
        c         of rows of a.
        c
        c       n is a positive integer input variable set to the number
        c         of columns of a.
        c
        c       a is an m by n array. on input a contains the matrix for
        c         which the qr factorization is to be computed. on output
        c         the strict upper trapezoidal part of a contains the strict
        c         upper trapezoidal part of r, and the lower trapezoidal
        c         part of a contains a factored form of q (the non-trivial
        c         elements of the u vectors described above).
        c
        c       lda is a positive integer input variable not less than m
        c         which specifies the leading dimension of the array a.
        c
        c       pivot is a logical input variable. if pivot is set true,
        c         then column pivoting is enforced. if pivot is set false,
        c         then no column pivoting is done.
        c
        c       ipvt is an integer output array of length lipvt. ipvt
        c         defines the permutation matrix p such that a*p = q*r.
        c         column j of p is column ipvt(j) of the identity matrix.
        c         if pivot is false, ipvt is not referenced.
        c
        c       lipvt is a positive integer input variable. if pivot is false,
        c         then lipvt may be as small as 1. if pivot is true, then
        c         lipvt must be at least n.
        c
        c       rdiag is an output array of length n which contains the
        c         diagonal elements of r.
        c
        c       acnorm is an output array of length n which contains the
        c         norms of the corresponding columns of the input matrix a.
        c         if this information is not needed, then acnorm can coincide
        c         with rdiag.
        c
        c       wa is a work array of length n. if pivot is false, then wa
        c         can coincide with rdiag.
        c
        c     subprograms called
        c
        c       minpack-supplied ... dpmpar,enorm
        c
        c       fortran-supplied ... dmax1,dsqrt,min0
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i, j, jp1, k, kmax, minmn;
        //      double ajnorm,one,p05,sum,temp,zero;
        double ajnorm, sum, temp;
        double fac;

        double tempvec[] = new double[m + 1];

        //      one = 1.0;
        //      p05 = .05;
        //      zero = 0.0;

        // Compute the initial column norms and initialize several arrays.

        for (j = 1; j <= n; j++) {

            for (i = 1; i <= m; i++) {

                tempvec[i] = a[i][j];

            }

            //         acnorm[j] = Minpack_f77.enorm_f77(m,a[1][j]);

            acnorm[j] = Minpack_f77.enorm_f77(m, tempvec);

            rdiag[j] = acnorm[j];
            wa[j] = rdiag[j];
            if (pivot)
                ipvt[j] = j;

        }

        // Reduce A to R with Householder transformations.

        minmn = Math.min(m, n);

        for (j = 1; j <= minmn; j++) {

            if (pivot) {

                // Bring the column of largest norm into the pivot position.

                kmax = j;

                for (k = j; k <= n; k++) {

                    if (rdiag[k] > rdiag[kmax])
                        kmax = k;

                }

                if (kmax != j) {

                    for (i = 1; i <= m; i++) {

                        temp = a[i][j];
                        a[i][j] = a[i][kmax];
                        a[i][kmax] = temp;

                    }

                    rdiag[kmax] = rdiag[j];
                    wa[kmax] = wa[j];
                    k = ipvt[j];
                    ipvt[j] = ipvt[kmax];
                    ipvt[kmax] = k;

                }

            }

            // Compute the Householder transformation to reduce the
            // j-th column of A to a multiple of the j-th unit vector.

            for (i = j; i <= m; i++) {

                tempvec[i - j + 1] = a[i][j];

            }

            //         ajnorm = Minpack_f77.enorm_f77(m-j+1,a[j][j]);
            ajnorm = Minpack_f77.enorm_f77(m - j + 1, tempvec);

            if (ajnorm != zero) {

                if (a[j][j] < zero)
                    ajnorm = -ajnorm;

                for (i = j; i <= m; i++) {

                    a[i][j] /= ajnorm;

                }

                a[j][j] += one;

                // Apply the transformation to the remaining columns
                // and update the norms.

                jp1 = j + 1;

                if (n >= jp1) {

                    for (k = jp1; k <= n; k++) {

                        sum = zero;

                        for (i = j; i <= m; i++) {

                            sum += a[i][j] * a[i][k];

                        }

                        temp = sum / a[j][j];

                        for (i = j; i <= m; i++) {

                            a[i][k] -= temp * a[i][j];

                        }

                        if (pivot && rdiag[k] != zero) {

                            temp = a[j][k] / rdiag[k];
                            rdiag[k] *= Math.sqrt(Math.max(zero, one - temp * temp));

                            fac = rdiag[k] / wa[k];
                            if (p05 * fac * fac <= epsmch) {

                                for (i = jp1; i <= m; i++) {

                                    tempvec[i - j] = a[i][k];

                                }

                                //                        rdiag[k] = Minpack_f77.enorm_f77(m-j,a[jp1][k]);
                                rdiag[k] = Minpack_f77.enorm_f77(m - j, tempvec);
                                wa[k] = rdiag[k];

                            }

                        }

                    }

                }

            }

            rdiag[j] = -ajnorm;

        }

        return;

    }

    /**
     * 
     *<p>
     * Given an m by n matrix A, an n by n diagonal matrix D, and an m-vector b,
     * the problem is to determine an x which solves the system
     * 
     * <pre>
     *    Ax = b ,     Dx = 0 ,
     *</pre>
     * 
     * in the least squares sense.
     *<p>
     * This method completes the solution of the problem if it is provided with
     * the necessary information from the QR factorization, with column
     * pivoting, of A. That is, if AP = QR, where P is a permutation matrix, Q
     * has orthogonal columns, and R is an upper triangular matrix with diagonal
     * elements of nonincreasing magnitude, then qrsolv_f77 expects the full
     * upper triangle of R, the permutation matrix P, and the first n components
     * of (Q transpose)b. The system
     * 
     * <pre>
     *           Ax = b, Dx = 0, is then equivalent to
     * 
     *                 t     t
     *           Rz = Q b,  P DPz = 0 ,
     *</pre>
     * 
     * where x = Pz. If this system does not have full rank, then a least
     * squares solution is obtained. On output qrsolv_f77 also provides an upper
     * triangular matrix S such that
     * 
     * <pre>
     *            t  t              t
     *           P (A A + DD)P = S S .
     *</pre>
     * 
     * S is computed within qrsolv_f77 and may be of separate interest.
     *<p>
     * Translated by Steve Verrill on November 17, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param n
     *            The order of r.
     *@param r
     *            r is an n by n array. On input the full upper triangle must
     *            contain the full upper triangle of the matrix R. On output the
     *            full upper triangle is unaltered, and the strict lower
     *            triangle contains the strict upper triangle (transposed) of
     *            the upper triangular matrix S.
     *@param ipvt
     *            ipvt is an integer input array of length n which defines the
     *            permutation matrix P such that AP = QR. Column j of P is
     *            column ipvt[j] of the identity matrix.
     *@param diag
     *            diag is an input array of length n which must contain the
     *            diagonal elements of the matrix D.
     *@param qtb
     *            qtb is an input array of length n which must contain the first
     *            n elements of the vector (Q transpose)b.
     *@param x
     *            x is an output array of length n which contains the least
     *            squares solution of the system Ax = b, Dx = 0.
     *@param sdiag
     *            sdiag is an output array of length n which contains the
     *            diagonal elements of the upper triangular matrix S.
     *@param wa
     *            wa is a work array of length n.
     * 
     * 
     */

    public static void qrsolv_f77(int n, double r[][], int ipvt[], double diag[], double qtb[], double x[],
            double sdiag[], double wa[]) {

        /*

        Here is a copy of the qrsolv FORTRAN documentation:


              subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
              integer n,ldr
              integer ipvt(n)
              double precision r(ldr,n),diag(n),qtb(n),x(n),sdiag(n),wa(n)

        c     **********
        c
        c     subroutine qrsolv
        c
        c     given an m by n matrix a, an n by n diagonal matrix d,
        c     and an m-vector b, the problem is to determine an x which
        c     solves the system
        c
        c           a*x = b ,     d*x = 0 ,
        c
        c     in the least squares sense.
        c
        c     this subroutine completes the solution of the problem
        c     if it is provided with the necessary information from the
        c     qr factorization, with column pivoting, of a. that is, if
        c     a*p = q*r, where p is a permutation matrix, q has orthogonal
        c     columns, and r is an upper triangular matrix with diagonal
        c     elements of nonincreasing magnitude, then qrsolv expects
        c     the full upper triangle of r, the permutation matrix p,
        c     and the first n components of (q transpose)*b. the system
        c     a*x = b, d*x = 0, is then equivalent to
        c
        c                  t       t
        c           r*z = q *b ,  p *d*p*z = 0 ,
        c
        c     where x = p*z. if this system does not have full rank,
        c     then a least squares solution is obtained. on output qrsolv
        c     also provides an upper triangular matrix s such that
        c
        c            t   t               t
        c           p *(a *a + d*d)*p = s *s .
        c
        c     s is computed within qrsolv and may be of separate interest.
        c
        c     the subroutine statement is
        c
        c       subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
        c
        c     where
        c
        c       n is a positive integer input variable set to the order of r.
        c
        c       r is an n by n array. on input the full upper triangle
        c         must contain the full upper triangle of the matrix r.
        c         on output the full upper triangle is unaltered, and the
        c         strict lower triangle contains the strict upper triangle
        c         (transposed) of the upper triangular matrix s.
        c
        c       ldr is a positive integer input variable not less than n
        c         which specifies the leading dimension of the array r.
        c
        c       ipvt is an integer input array of length n which defines the
        c         permutation matrix p such that a*p = q*r. column j of p
        c         is column ipvt(j) of the identity matrix.
        c
        c       diag is an input array of length n which must contain the
        c         diagonal elements of the matrix d.
        c
        c       qtb is an input array of length n which must contain the first
        c         n elements of the vector (q transpose)*b.
        c
        c       x is an output array of length n which contains the least
        c         squares solution of the system a*x = b, d*x = 0.
        c
        c       sdiag is an output array of length n which contains the
        c         diagonal elements of the upper triangular matrix s.
        c
        c       wa is a work array of length n.
        c
        c     subprograms called
        c
        c       fortran-supplied ... dabs,dsqrt
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i, j, jp1, k, kp1, l, nsing;
        //      double cos,cotan,p5,p25,qtbpj,sin,sum,tan,temp,zero;
        double cos, cotan, qtbpj, sin, sum, tan, temp;

        //      p5 = .5;
        //      p25 = .25;
        //      zero = 0.0;

        // Copy R and (Q transpose)b to preserve input and initialize S.
        // In particular, save the diagonal elements of R in x.

        for (j = 1; j <= n; j++) {

            for (i = j; i <= n; i++) {

                r[i][j] = r[j][i];

            }

            x[j] = r[j][j];
            wa[j] = qtb[j];

        }

        // Eliminate the diagonal matrix D using a Givens rotation.

        for (j = 1; j <= n; j++) {

            // Prepare the row of D to be eliminated, locating the
            // diagonal element using P from the QR factorization.

            l = ipvt[j];

            if (diag[l] != zero) {

                for (k = j; k <= n; k++) {

                    sdiag[k] = zero;

                }

                sdiag[j] = diag[l];

                // The transformations to eliminate the row of D
                // modify only a single element of (Q transpose)b
                // beyond the first n, which is initially zero.    ??????

                qtbpj = zero;

                for (k = j; k <= n; k++) {

                    // Determine a Givens rotation which eliminates the
                    // appropriate element in the current row of D.

                    if (sdiag[k] != zero) {

                        if (Math.abs(r[k][k]) < Math.abs(sdiag[k])) {

                            cotan = r[k][k] / sdiag[k];
                            sin = p5 / Math.sqrt(p25 + p25 * cotan * cotan);
                            cos = sin * cotan;

                        } else {

                            tan = sdiag[k] / r[k][k];
                            cos = p5 / Math.sqrt(p25 + p25 * tan * tan);
                            sin = cos * tan;

                        }

                        // Compute the modified diagonal element of R and
                        // the modified element of ((Q transpose)b,0).

                        r[k][k] = cos * r[k][k] + sin * sdiag[k];
                        temp = cos * wa[k] + sin * qtbpj;
                        qtbpj = -sin * wa[k] + cos * qtbpj;
                        wa[k] = temp;

                        // Accumulate the tranformation in the row of S.

                        kp1 = k + 1;

                        for (i = kp1; i <= n; i++) {

                            temp = cos * r[i][k] + sin * sdiag[i];
                            sdiag[i] = -sin * r[i][k] + cos * sdiag[i];
                            r[i][k] = temp;

                        }

                    }

                }

            }

            // Store the diagonal element of S and restore
            // the corresponding diagonal element of R.

            sdiag[j] = r[j][j];
            r[j][j] = x[j];

        }

        // Solve the triangular system for z. if the system is
        // singular, then obtain a least squares solution.

        nsing = n;

        for (j = 1; j <= n; j++) {

            if (sdiag[j] == zero && nsing == n)
                nsing = j - 1;
            if (nsing < n)
                wa[j] = zero;

        }

        //      if (nsing >= 1) {

        for (k = 1; k <= nsing; k++) {

            j = nsing - k + 1;
            sum = zero;
            jp1 = j + 1;

            //         if (nsing >= jp1) {

            for (i = jp1; i <= nsing; i++) {

                sum += r[i][j] * wa[i];

            }

            //         }

            wa[j] = (wa[j] - sum) / sdiag[j];

        }

        //      }

        // Permute the components of z back to components of x.

        for (j = 1; j <= n; j++) {

            l = ipvt[j];
            x[l] = wa[j];

        }

        return;

    }

    /**
     * 
     *<p>
     * Given an m by n matrix A, an n by n nonsingular diagonal matrix D, an
     * m-vector b, and a positive number delta, the problem is to determine a
     * value for the parameter par such that if x solves the system
     * 
     * <pre>
     *           A*x = b ,     sqrt(par)*D*x = 0
     *</pre>
     * 
     * in the least squares sense, and dxnorm is the Euclidean norm of D*x, then
     * either par is zero and
     * 
     * <pre>
     *           (dxnorm-delta) &lt;= 0.1*delta ,
     *</pre>
     * 
     * or par is positive and
     * 
     * <pre>
     *           abs(dxnorm-delta) &lt;= 0.1*delta .
     *</pre>
     * 
     * This method (lmpar_f77) completes the solution of the problem if it is
     * provided with the necessary information from the QR factorization, with
     * column pivoting, of A. That is, if AP = QR, where P is a permutation
     * matrix, Q has orthogonal columns, and R is an upper triangular matrix
     * with diagonal elements of nonincreasing magnitude, then lmpar_f77 expects
     * the full upper triangle of R, the permutation matrix P, and the first n
     * components of (Q transpose)b. On output lmpar_f77 also provides an upper
     * triangular matrix S such that
     * 
     * <pre>
     *            t  t                t
     *           P (A A + par*DD)P = S S .
     *</pre>
     * 
     * S is employed within lmpar_f77 and may be of separate interest.
     *<p>
     * Only a few iterations are generally needed for convergence of the
     * algorithm. If, however, the limit of 10 iterations is reached, then the
     * output par will contain the best value obtained so far.
     *<p>
     * Translated by Steve Verrill on November 17, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param n
     *            The order of r.
     *@param r
     *            r is an n by n array. On input the full upper triangle must
     *            contain the full upper triangle of the matrix R. On output the
     *            full upper triangle is unaltered, and the strict lower
     *            triangle contains the strict upper triangle (transposed) of
     *            the upper triangular matrix S.
     *@param ipvt
     *            ipvt is an integer input array of length n which defines the
     *            permutation matrix P such that AP = QR. Column j of P is
     *            column ipvt[j] of the identity matrix.
     *@param diag
     *            diag is an input array of length n which must contain the
     *            diagonal elements of the matrix D.
     *@param qtb
     *            qtb is an input array of length n which must contain the first
     *            n elements of the vector (Q transpose)b.
     *@param delta
     *            delta is a positive input variable which specifies an upper
     *            bound on the Euclidean norm of Dx.
     *@param par
     *            par is a nonnegative variable. On input par contains an
     *            initial estimate of the Levenberg-Marquardt parameter. On
     *            output par contains the final estimate.
     *@param x
     *            x is an output array of length n which contains the least
     *            squares solution of the system Ax = b, sqrt(par)*Dx = 0, for
     *            the output par.
     *@param sdiag
     *            sdiag is an output array of length n which contains the
     *            diagonal elements of the upper triangular matrix S.
     *@param wa1
     *            wa1 is a work array of length n.
     *@param wa2
     *            wa2 is a work array of length n.
     * 
     * 
     */

    public static void lmpar_f77(int n, double r[][], int ipvt[], double diag[], double qtb[], double delta,
            double par[], double x[], double sdiag[], double wa1[], double wa2[]) {

        /*

        Here is a copy of the lmpar FORTRAN documentation:


              subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,wa1,
             *                 wa2)
              integer n,ldr
              integer ipvt(n)
              double precision delta,par
              double precision r(ldr,n),diag(n),qtb(n),x(n),sdiag(n),wa1(n),
             *                 wa2(n)

        c     **********
        c
        c     subroutine lmpar
        c
        c     given an m by n matrix a, an n by n nonsingular diagonal
        c     matrix d, an m-vector b, and a positive number delta,
        c     the problem is to determine a value for the parameter
        c     par such that if x solves the system
        c
        c           a*x = b ,     sqrt(par)*d*x = 0 ,
        c
        c     in the least squares sense, and dxnorm is the euclidean
        c     norm of d*x, then either par is zero and
        c
        c           (dxnorm-delta) .le. 0.1*delta ,
        c
        c     or par is positive and
        c
        c           abs(dxnorm-delta) .le. 0.1*delta .
        c
        c     this subroutine completes the solution of the problem
        c     if it is provided with the necessary information from the
        c     qr factorization, with column pivoting, of a. that is, if
        c     a*p = q*r, where p is a permutation matrix, q has orthogonal
        c     columns, and r is an upper triangular matrix with diagonal
        c     elements of nonincreasing magnitude, then lmpar expects
        c     the full upper triangle of r, the permutation matrix p,
        c     and the first n components of (q transpose)*b. on output
        c     lmpar also provides an upper triangular matrix s such that
        c
        c            t   t                   t
        c           p *(a *a + par*d*d)*p = s *s .
        c
        c     s is employed within lmpar and may be of separate interest.
        c
        c     only a few iterations are generally needed for convergence
        c     of the algorithm. if, however, the limit of 10 iterations
        c     is reached, then the output par will contain the best
        c     value obtained so far.
        c
        c     the subroutine statement is
        c
        c       subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,
        c                        wa1,wa2)
        c
        c     where
        c
        c       n is a positive integer input variable set to the order of r.
        c
        c       r is an n by n array. on input the full upper triangle
        c         must contain the full upper triangle of the matrix r.
        c         on output the full upper triangle is unaltered, and the
        c         strict lower triangle contains the strict upper triangle
        c         (transposed) of the upper triangular matrix s.
        c
        c       ldr is a positive integer input variable not less than n
        c         which specifies the leading dimension of the array r.
        c
        c       ipvt is an integer input array of length n which defines the
        c         permutation matrix p such that a*p = q*r. column j of p
        c         is column ipvt(j) of the identity matrix.
        c
        c       diag is an input array of length n which must contain the
        c         diagonal elements of the matrix d.
        c
        c       qtb is an input array of length n which must contain the first
        c         n elements of the vector (q transpose)*b.
        c
        c       delta is a positive input variable which specifies an upper
        c         bound on the euclidean norm of d*x.
        c
        c       par is a nonnegative variable. on input par contains an
        c         initial estimate of the levenberg-marquardt parameter.
        c         on output par contains the final estimate.
        c
        c       x is an output array of length n which contains the least
        c         squares solution of the system a*x = b, sqrt(par)*d*x = 0,
        c         for the output par.
        c
        c       sdiag is an output array of length n which contains the
        c         diagonal elements of the upper triangular matrix s.
        c
        c       wa1 and wa2 are work arrays of length n.
        c
        c     subprograms called
        c
        c       minpack-supplied ... dpmpar,enorm,qrsolv
        c
        c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i, iter, j, jm1, jp1, k, l, nsing;

        //      double dxnorm,dwarf,fp,gnorm,parc,parl,paru,p1,p001,
        //             sum,temp,zero;
        double dxnorm, dwarf, fp, gnorm, parc, parl, paru, sum, temp;

        boolean loop;

        //      p1 = .1;
        //      p001 = .001;
        //      zero = 0.0;

        // dwarf is the smallest positive magnitude.

        dwarf = minmag;

        // Compute and store in x the Gauss-Newton direction.  If the
        // Jacobian is rank-deficient, obtain a least squares solution.

        nsing = n;

        for (j = 1; j <= n; j++) {

            wa1[j] = qtb[j];
            if (r[j][j] == zero && nsing == n)
                nsing = j - 1;
            if (nsing < n)
                wa1[j] = zero;

        }

        //      if (nsing >= 1) {

        for (k = 1; k <= nsing; k++) {

            j = nsing - k + 1;
            wa1[j] /= r[j][j];
            temp = wa1[j];
            jm1 = j - 1;

            //         if (jm1 >= 1) {

            for (i = 1; i <= jm1; i++) {

                wa1[i] -= r[i][j] * temp;

            }

            //         }

        }

        //      }

        for (j = 1; j <= n; j++) {

            l = ipvt[j];
            x[l] = wa1[j];

        }

        // Initialize the iteration counter.
        // Evaluate the function at the origin, and test
        // for acceptance of the Gauss-Newton direction.

        iter = 0;

        for (j = 1; j <= n; j++) {

            wa2[j] = diag[j] * x[j];

        }

        dxnorm = Minpack_f77.enorm_f77(n, wa2);

        fp = dxnorm - delta;

        if (fp <= p1 * delta) {

            par[1] = zero;
            return;

        }

        // If the Jacobian is not rank deficient, the Newton
        // step provides a lower bound, parl, for the zero of
        // the function.  Otherwise set this bound to zero.

        parl = zero;

        if (nsing >= n) {

            for (j = 1; j <= n; j++) {

                l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);

            }

            for (j = 1; j <= n; j++) {

                sum = zero;
                jm1 = j - 1;

                //            if (jm1 >= 1) {

                for (i = 1; i <= jm1; i++) {

                    sum += r[i][j] * wa1[i];

                }

                //            }

                wa1[j] = (wa1[j] - sum) / r[j][j];

            }

            temp = Minpack_f77.enorm_f77(n, wa1);
            parl = ((fp / delta) / temp) / temp;

        }

        // Calculate an upper bound, paru, for the zero of the function.

        for (j = 1; j <= n; j++) {

            sum = zero;

            for (i = 1; i <= j; i++) {

                sum += r[i][j] * qtb[i];

            }

            l = ipvt[j];
            wa1[j] = sum / diag[l];

        }

        gnorm = Minpack_f77.enorm_f77(n, wa1);
        paru = gnorm / delta;

        if (paru == zero)
            paru = dwarf / Math.min(delta, p1);

        // If the input par lies outside of the interval (parl,paru),
        // set par to the closer endpoint.

        par[1] = Math.max(par[1], parl);
        par[1] = Math.min(par[1], paru);

        if (par[1] == zero)
            par[1] = gnorm / dxnorm;

        // Beginning of an iteration.

        loop = true;

        while (loop) {

            iter++;

            // Evaluate the function at the current value of par.

            if (par[1] == zero)
                par[1] = Math.max(dwarf, p001 * paru);
            temp = Math.sqrt(par[1]);

            for (j = 1; j <= n; j++) {

                wa1[j] = temp * diag[j];

            }

            Minpack_f77.qrsolv_f77(n, r, ipvt, wa1, qtb, x, sdiag, wa2);

            for (j = 1; j <= n; j++) {

                wa2[j] = diag[j] * x[j];

            }

            dxnorm = Minpack_f77.enorm_f77(n, wa2);
            temp = fp;
            fp = dxnorm - delta;

            // If the function is small enough, accept the current value
            // of par.  Also test for the exceptional cases where parl
            // is zero or the number of iterations has reached 10.

            if (Math.abs(fp) <= p1 * delta || parl == zero && fp <= temp && temp < zero || iter == 10) {

                // Termination

                if (iter == 0)
                    par[1] = zero;
                return;

            }

            // Compute the Newton correction.

            for (j = 1; j <= n; j++) {

                l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);

            }

            for (j = 1; j <= n; j++) {

                wa1[j] /= sdiag[j];
                temp = wa1[j];
                jp1 = j + 1;

                for (i = jp1; i <= n; i++) {

                    wa1[i] -= r[i][j] * temp;

                }

            }

            temp = Minpack_f77.enorm_f77(n, wa1);
            parc = ((fp / delta) / temp) / temp;

            // Depending on the sign of the function, update parl or paru.

            if (fp > zero)
                parl = Math.max(parl, par[1]);
            if (fp < zero)
                paru = Math.min(paru, par[1]);

            // Compute an improved estimate for par[1].

            par[1] = Math.max(parl, par[1] + parc);

            // End of an iteration.

        }

    }

    /**
     * 
     *<p>
     * The lmdif1_f77 method minimizes the sum of the squares of m nonlinear
     * functions in n variables by a modification of the Levenberg-Marquardt
     * algorithm. This is done by using the more general least-squares solver
     * lmdif. The user must provide a method that calculates the functions. The
     * Jacobian is then calculated by a forward-difference approximation.
     *<p>
     * Translated by Steve Verrill on November 24, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param nlls
     *            A class that implements the Lmdif_fcn interface (see the
     *            definition in Lmdif_fcn.java). See LmdifTest_f77.java for an
     *            example of such a class. The class must define a method, fcn,
     *            that must have the form
     * 
     *            public static void fcn(int m, int n, double x[], double
     *            fvec[], int iflag[])
     * 
     *            The value of iflag[1] should not be changed by fcn unless the
     *            user wants to terminate execution of lmdif_f77. In this case
     *            set iflag[1] to a negative integer.
     * 
     *@param m
     *            A positive integer set to the number of functions [number of
     *            observations]
     *@param n
     *            A positive integer set to the number of variables [number of
     *            parameters]. n must not exceed m.
     *@param x
     *            On input, it contains the initial estimate of the solution
     *            vector [the least squares parameters]. On output it contains
     *            the final estimate of the solution vector.
     *@param fvec
     *            An output vector that contains the m functions [residuals]
     *            evaluated at x.
     *@param tol
     *            tol is a nonnegative input variable. Termination occurs when
     *            the algorithm estimates either that the relative error in the
     *            sum of squares is at most tol or that the relative error
     *            between x and the solution is at most tol.
     *@param info
     *            An integer output variable. If the user has terminated
     *            execution, info is set to the (negative) value of iflag[1].
     *            See description of fcn. Otherwise, info is set as follows.
     * 
     *            info = 0 improper input parameters.
     * 
     *            info = 1 algorithm estimates that the relative error in the
     *            sum of squares is at most tol.
     * 
     *            info = 2 algorithm estimates that the relative error between x
     *            and the solution is at most tol.
     * 
     *            info = 3 conditions for info = 1 and info = 2 both hold.
     * 
     *            info = 4 fvec is orthogonal to the columns of the Jacobian to
     *            machine precision.
     * 
     *            info = 5 number of calls to fcn has reached or exceeded
     *            200*(n+1).
     * 
     *            info = 6 tol is too small. No further reduction in the sum of
     *            squares is possible.
     * 
     *            info = 7 tol is too small. No further improvement in the
     *            approximate solution x is possible.
     * 
     */

    public static void lmdif1_f77(Lmdif_fcn nlls, int m, int n, double x[], double fvec[], double tol, int info[]) {

        /*

        Here is a copy of the lmdif1 FORTRAN documentation:


              subroutine lmdif1(fcn,m,n,x,fvec,tol,info,iwa,wa,lwa)
              integer m,n,info,lwa
              integer iwa(n)
              double precision tol
              double precision x(n),fvec(m),wa(lwa)
              external fcn
        c     **********
        c
        c     subroutine lmdif1
        c
        c     the purpose of lmdif1 is to minimize the sum of the squares of
        c     m nonlinear functions in n variables by a modification of the
        c     levenberg-marquardt algorithm. this is done by using the more
        c     general least-squares solver lmdif. the user must provide a
        c     subroutine which calculates the functions. the jacobian is
        c     then calculated by a forward-difference approximation.
        c
        c     the subroutine statement is
        c
        c       subroutine lmdif1(fcn,m,n,x,fvec,tol,info,iwa,wa,lwa)
        c
        c     where
        c
        c       fcn is the name of the user-supplied subroutine which
        c         calculates the functions. fcn must be declared
        c         in an external statement in the user calling
        c         program, and should be written as follows.
        c
        c         subroutine fcn(m,n,x,fvec,iflag)
        c         integer m,n,iflag
        c         double precision x(n),fvec(m)
        c         ----------
        c         calculate the functions at x and
        c         return this vector in fvec.
        c         ----------
        c         return
        c         end
        c
        c         the value of iflag should not be changed by fcn unless
        c         the user wants to terminate execution of lmdif1.
        c         in this case set iflag to a negative integer.
        c
        c       m is a positive integer input variable set to the number
        c         of functions.
        c
        c       n is a positive integer input variable set to the number
        c         of variables. n must not exceed m.
        c
        c       x is an array of length n. on input x must contain
        c         an initial estimate of the solution vector. on output x
        c         contains the final estimate of the solution vector.
        c
        c       fvec is an output array of length m which contains
        c         the functions evaluated at the output x.
        c
        c       tol is a nonnegative input variable. termination occurs
        c         when the algorithm estimates either that the relative
        c         error in the sum of squares is at most tol or that
        c         the relative error between x and the solution is at
        c         most tol.
        c
        c       info is an integer output variable. if the user has
        c         terminated execution, info is set to the (negative)
        c         value of iflag. see description of fcn. otherwise,
        c         info is set as follows.
        c
        c         info = 0  improper input parameters.
        c
        c         info = 1  algorithm estimates that the relative error
        c                   in the sum of squares is at most tol.
        c
        c         info = 2  algorithm estimates that the relative error
        c                   between x and the solution is at most tol.
        c
        c         info = 3  conditions for info = 1 and info = 2 both hold.
        c
        c         info = 4  fvec is orthogonal to the columns of the
        c                   jacobian to machine precision.
        c
        c         info = 5  number of calls to fcn has reached or
        c                   exceeded 200*(n+1).
        c
        c         info = 6  tol is too small. no further reduction in
        c                   the sum of squares is possible.
        c
        c         info = 7  tol is too small. no further improvement in
        c                   the approximate solution x is possible.
        c
        c       iwa is an integer work array of length n.
        c
        c       wa is a work array of length lwa.
        c
        c       lwa is a positive integer input variable not less than
        c         m*n+5*n+m.
        c
        c     subprograms called
        c
        c       user-supplied ...... fcn
        c
        c       minpack-supplied ... lmdif
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int maxfev, mode, nprint;

        //      double epsfcn,factor,ftol,gtol,xtol,zero;
        double epsfcn, factor, ftol, gtol, xtol;

        double diag[] = new double[n + 1];
        int nfev[] = new int[2];
        double fjac[][] = new double[m + 1][n + 1];
        int ipvt[] = new int[n + 1];
        double qtf[] = new double[n + 1];

        factor = 100.0;
        //      zero = 0.0;

        info[1] = 0;

        // Check the input parameters for errors.

        if (n <= 0 || m < n || tol < zero) {

            return;

        }

        // Call lmdif.

        maxfev = 200 * (n + 1);
        ftol = tol;
        xtol = tol;
        gtol = zero;
        epsfcn = zero;
        mode = 1;
        nprint = 0;

        Minpack_f77.lmdif_f77(nlls, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor, nprint, info,
                nfev, fjac, ipvt, qtf);

        if (info[1] == 8)
            info[1] = 4;

        return;

    }

    /**
     * 
     *<p>
     * The lmdif_f77 method minimizes the sum of the squares of m nonlinear
     * functions in n variables by a modification of the Levenberg-Marquardt
     * algorithm. The user must provide a method that calculates the functions.
     * The Jacobian is then calculated by a forward-difference approximation.
     *<p>
     * Translated by Steve Verrill on November 20, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param nlls
     *            A class that implements the Lmdif_fcn interface (see the
     *            definition in Lmdif_fcn.java). See LmdifTest_f77.java for an
     *            example of such a class. The class must define a method, fcn,
     *            that must have the form
     * 
     *            public static void fcn(int m, int n, double x[], double
     *            fvec[], int iflag[])
     * 
     *            The value of iflag[1] should not be changed by fcn unless the
     *            user wants to terminate execution of lmdif_f77. In this case
     *            set iflag[1] to a negative integer.
     * 
     *@param m
     *            A positive integer set to the number of functions [number of
     *            observations]
     *@param n
     *            A positive integer set to the number of variables [number of
     *            parameters]. n must not exceed m.
     *@param x
     *            On input, it contains the initial estimate of the solution
     *            vector [the least squares parameters]. On output it contains
     *            the final estimate of the solution vector.
     *@param fvec
     *            An output vector that contains the m functions [residuals]
     *            evaluated at x.
     *@param ftol
     *            A nonnegative input variable. Termination occurs when both the
     *            actual and predicted relative reductions in the sum of squares
     *            are at most ftol. Therefore, ftol measures the relative error
     *            desired in the sum of squares.
     *@param xtol
     *            A nonnegative input variable. Termination occurs when the
     *            relative error between two consecutive iterates is at most
     *            xtol. Therefore, xtol measures the relative error desired in
     *            the approximate solution.
     *@param gtol
     *            A nonnegative input variable. Termination occurs when the
     *            cosine of the angle between fvec and any column of the
     *            Jacobian is at most gtol in absolute value. Therefore, gtol
     *            measures the orthogonality desired between the function vector
     *            and the columns of the Jacobian.
     *@param maxfev
     *            A positive integer input variable. Termination occurs when the
     *            number of calls to fcn is at least maxfev by the end of an
     *            iteration.
     *@param epsfcn
     *            An input variable used in determining a suitable step length
     *            for the forward-difference approximation. This approximation
     *            assumes that the relative errors in the functions are of the
     *            order of epsfcn. If epsfcn is less than the machine precision,
     *            it is assumed that the relative errors in the functions are of
     *            the order of the machine precision.
     *@param diag
     *            An vector of length n. If mode = 1 (see below), diag is
     *            internally set. If mode = 2, diag must contain positive
     *            entries that serve as multiplicative scale factors for the
     *            variables.
     *@param mode
     *            If mode = 1, the variables will be scaled internally. If mode
     *            = 2, the scaling is specified by the input diag. Other values
     *            of mode are equivalent to mode = 1.
     *@param factor
     *            A positive input variable used in determining the initial step
     *            bound. This bound is set to the product of factor and the
     *            euclidean norm of diag*x if nonzero, or else to factor itself.
     *            In most cases factor should lie in the interval (.1,100). 100
     *            is a generally recommended value.
     *@param nprint
     *            An integer input variable that enables controlled printing of
     *            iterates if it is positive. In this case, fcn is called with
     *            iflag[1] = 0 at the beginning of the first iteration and every
     *            nprint iterations thereafter and immediately prior to return,
     *            with x and fvec available for printing. If nprint is not
     *            positive, no special calls of fcn with iflag[1] = 0 are made.
     *@param info
     *            An integer output variable. If the user has terminated
     *            execution, info is set to the (negative) value of iflag[1].
     *            See description of fcn. Otherwise, info is set as follows.
     * 
     *            info = 0 improper input parameters.
     * 
     *            info = 1 both actual and predicted relative reductions in the
     *            sum of squares are at most ftol.
     * 
     *            info = 2 relative error between two consecutive iterates is at
     *            most xtol.
     * 
     *            info = 3 conditions for info = 1 and info = 2 both hold.
     * 
     *            info = 4 the cosine of the angle between fvec and any column
     *            of the Jacobian is at most gtol in absolute value.
     * 
     *            info = 5 number of calls to fcn with iflag[1] = 1 has reached
     *            maxfev.
     * 
     *            info = 6 ftol is too small. no further reduction in the sum of
     *            squares is possible.
     * 
     *            info = 7 xtol is too small. no further improvement in the
     *            approximate solution x is possible.
     * 
     *            info = 8 gtol is too small. fvec is orthogonal to the columns
     *            of the Jacobian to machine precision.
     * 
     *@param nfev
     *            An integer output variable set to the number of calls to fcn.
     *@param fjac
     *            An output m by n array. The upper n by n submatrix of fjac
     *            contains an upper triangular matrix R with diagonal elements
     *            of nonincreasing magnitude such that
     * 
     *            t t t P (jac *jac)P = R R,
     * 
     *            where P is a permutation matrix and jac is the final
     *            calculated Jacobian. Column j of P is column ipvt[j] (see
     *            below) of the identity matrix. The lower trapezoidal part of
     *            fjac contains information generated during the computation of
     *            R.
     *@param ipvt
     *            An integer output array of length n. ipvt defines a
     *            permutation matrix P such that jac*P = QR, where jac is the
     *            final calculated Jacobian, Q is orthogonal (not stored), and R
     *            is upper triangular with diagonal elements of nonincreasing
     *            magnitude. column j of P is column ipvt[j] of the identity
     *            matrix.
     * 
     *@param qtf
     *            An output array of length n which contains the first n
     *            elements of the vector (Q transpose)fvec.
     * 
     * 
     */

    public static void lmdif_f77(Lmdif_fcn nlls, int m, int n, double x[], double fvec[], double ftol, double xtol,
            double gtol, int maxfev, double epsfcn, double diag[], int mode, double factor, int nprint, int info[],
            int nfev[], double fjac[][], int ipvt[], double qtf[]) {

        /*

        Here is a copy of the lmdif FORTRAN documentation:


              subroutine lmdif(fcn,m,n,x,fvec,ftol,xtol,gtol,maxfev,epsfcn,
             *                 diag,mode,factor,nprint,info,nfev,fjac,ldfjac,
             *                 ipvt,qtf,wa1,wa2,wa3,wa4)
              integer m,n,maxfev,mode,nprint,info,nfev,ldfjac
              integer ipvt(n)
              double precision ftol,xtol,gtol,epsfcn,factor
              double precision x(n),fvec(m),diag(n),fjac(ldfjac,n),qtf(n),
             *                 wa1(n),wa2(n),wa3(n),wa4(m)
              external fcn
        c     **********
        c
        c     subroutine lmdif
        c
        c     the purpose of lmdif is to minimize the sum of the squares of
        c     m nonlinear functions in n variables by a modification of
        c     the levenberg-marquardt algorithm. the user must provide a
        c     subroutine which calculates the functions. the jacobian is
        c     then calculated by a forward-difference approximation.
        c
        c     the subroutine statement is
        c
        c       subroutine lmdif(fcn,m,n,x,fvec,ftol,xtol,gtol,maxfev,epsfcn,
        c                        diag,mode,factor,nprint,info,nfev,fjac,
        c                        ldfjac,ipvt,qtf,wa1,wa2,wa3,wa4)
        c
        c     where
        c
        c       fcn is the name of the user-supplied subroutine which
        c         calculates the functions. fcn must be declared
        c         in an external statement in the user calling
        c         program, and should be written as follows.
        c
        c         subroutine fcn(m,n,x,fvec,iflag)
        c         integer m,n,iflag
        c         double precision x(n),fvec(m)
        c         ----------
        c         calculate the functions at x and
        c         return this vector in fvec.
        c         ----------
        c         return
        c         end
        c
        c         the value of iflag should not be changed by fcn unless
        c         the user wants to terminate execution of lmdif.
        c         in this case set iflag to a negative integer.
        c
        c       m is a positive integer input variable set to the number
        c         of functions.
        c
        c       n is a positive integer input variable set to the number
        c         of variables. n must not exceed m.
        c
        c       x is an array of length n. on input x must contain
        c         an initial estimate of the solution vector. on output x
        c         contains the final estimate of the solution vector.
        c
        c       fvec is an output array of length m which contains
        c         the functions evaluated at the output x.
        c
        c       ftol is a nonnegative input variable. termination
        c         occurs when both the actual and predicted relative
        c         reductions in the sum of squares are at most ftol.
        c         therefore, ftol measures the relative error desired
        c         in the sum of squares.
        c
        c       xtol is a nonnegative input variable. termination
        c         occurs when the relative error between two consecutive
        c         iterates is at most xtol. therefore, xtol measures the
        c         relative error desired in the approximate solution.
        c
        c       gtol is a nonnegative input variable. termination
        c         occurs when the cosine of the angle between fvec and
        c         any column of the jacobian is at most gtol in absolute
        c         value. therefore, gtol measures the orthogonality
        c         desired between the function vector and the columns
        c         of the jacobian.
        c
        c       maxfev is a positive integer input variable. termination
        c         occurs when the number of calls to fcn is at least
        c         maxfev by the end of an iteration.
        c
        c       epsfcn is an input variable used in determining a suitable
        c         step length for the forward-difference approximation. this
        c         approximation assumes that the relative errors in the
        c         functions are of the order of epsfcn. if epsfcn is less
        c         than the machine precision, it is assumed that the relative
        c         errors in the functions are of the order of the machine
        c         precision.
        c
        c       diag is an array of length n. if mode = 1 (see
        c         below), diag is internally set. if mode = 2, diag
        c         must contain positive entries that serve as
        c         multiplicative scale factors for the variables.
        c
        c       mode is an integer input variable. if mode = 1, the
        c         variables will be scaled internally. if mode = 2,
        c         the scaling is specified by the input diag. other
        c         values of mode are equivalent to mode = 1.
        c
        c       factor is a positive input variable used in determining the
        c         initial step bound. this bound is set to the product of
        c         factor and the euclidean norm of diag*x if nonzero, or else
        c         to factor itself. in most cases factor should lie in the
        c         interval (.1,100.). 100. is a generally recommended value.
        c
        c       nprint is an integer input variable that enables controlled
        c         printing of iterates if it is positive. in this case,
        c         fcn is called with iflag = 0 at the beginning of the first
        c         iteration and every nprint iterations thereafter and
        c         immediately prior to return, with x and fvec available
        c         for printing. if nprint is not positive, no special calls
        c         of fcn with iflag = 0 are made.
        c
        c       info is an integer output variable. if the user has
        c         terminated execution, info is set to the (negative)
        c         value of iflag. see description of fcn. otherwise,
        c         info is set as follows.
        c
        c         info = 0  improper input parameters.
        c
        c         info = 1  both actual and predicted relative reductions
        c                   in the sum of squares are at most ftol.
        c
        c         info = 2  relative error between two consecutive iterates
        c                   is at most xtol.
        c
        c         info = 3  conditions for info = 1 and info = 2 both hold.
        c
        c         info = 4  the cosine of the angle between fvec and any
        c                   column of the jacobian is at most gtol in
        c                   absolute value.
        c
        c         info = 5  number of calls to fcn has reached or
        c                   exceeded maxfev.
        c
        c         info = 6  ftol is too small. no further reduction in
        c                   the sum of squares is possible.
        c
        c         info = 7  xtol is too small. no further improvement in
        c                   the approximate solution x is possible.
        c
        c         info = 8  gtol is too small. fvec is orthogonal to the
        c                   columns of the jacobian to machine precision.
        c
        c       nfev is an integer output variable set to the number of
        c         calls to fcn.
        c
        c       fjac is an output m by n array. the upper n by n submatrix
        c         of fjac contains an upper triangular matrix r with
        c         diagonal elements of nonincreasing magnitude such that
        c
        c                t     t           t
        c               p *(jac *jac)*p = r *r,
        c
        c         where p is a permutation matrix and jac is the final
        c         calculated jacobian. column j of p is column ipvt(j)
        c         (see below) of the identity matrix. the lower trapezoidal
        c         part of fjac contains information generated during
        c         the computation of r.
        c
        c       ldfjac is a positive integer input variable not less than m
        c         which specifies the leading dimension of the array fjac.
        c
        c       ipvt is an integer output array of length n. ipvt
        c         defines a permutation matrix p such that jac*p = q*r,
        c         where jac is the final calculated jacobian, q is
        c         orthogonal (not stored), and r is upper triangular
        c         with diagonal elements of nonincreasing magnitude.
        c         column j of p is column ipvt(j) of the identity matrix.
        c
        c       qtf is an output array of length n which contains
        c         the first n elements of the vector (q transpose)*fvec.
        c
        c       wa1, wa2, and wa3 are work arrays of length n.
        c
        c       wa4 is a work array of length m.
        c
        c     subprograms called
        c
        c       user-supplied ...... fcn
        c
        c       minpack-supplied ... dpmpar,enorm,fdjac2,lmpar,qrfac
        c
        c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt,mod
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i, iter, j, l;

        //      double actred,delta,dirder,fnorm,fnorm1,gnorm,
        //             one,pnorm,prered,p1,p5,p25,p75,p0001,ratio,
        //             sum,temp,temp1,temp2,xnorm,zero;
        double actred, delta, dirder, fnorm, fnorm1, gnorm, pnorm, prered, ratio, sum, temp, temp1, temp2, xnorm;

        double par[] = new double[2];

        boolean doneout, donein;

        int iflag[] = new int[2];
        double wa1[] = new double[n + 1];
        double wa2[] = new double[n + 1];
        double wa3[] = new double[n + 1];
        double wa4[] = new double[m + 1];

        // The Java compiler complaines if delta and xnorm have not been
        // initialized.

        delta = 0.0;
        xnorm = 0.0;

        //      one = 1.0;
        //      p1 = .1;
        //      p25 = .25;
        //      p5 = .5;
        //      p75 = .75;
        //      p0001 = .0001;
        //      zero = 0.0;

        info[1] = 0;
        iflag[1] = 0;
        nfev[1] = 0;

        // Check the input parameters for errors.

        if (n <= 0 || m < n || ftol < zero || xtol < zero || gtol < zero || maxfev <= 0 || factor <= zero) {

            // Termination 

            if (nprint > 0) {

                nlls.fcn(m, n, x, fvec, iflag);

            }

            return;

        }

        if (mode == 2) {

            for (j = 1; j <= n; j++) {

                if (diag[j] <= zero) {

                    // Termination

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, iflag);

                    }

                    return;

                }

            }

        }

        // Evaluate the function at the starting point
        // and calculate its norm.

        iflag[1] = 1;

        nlls.fcn(m, n, x, fvec, iflag);

        nfev[1] = 1;

        if (iflag[1] < 0) {

            // Termination

            info[1] = iflag[1];
            iflag[1] = 0;

            if (nprint > 0) {

                nlls.fcn(m, n, x, fvec, iflag);

            }

            return;

        }

        fnorm = Minpack_f77.enorm_f77(m, fvec);

        // Initialize Levenberg-Marquardt parameter and iteration counter.

        par[1] = zero;
        iter = 1;

        // Beginning of the outer loop.

        doneout = false;

        while (!doneout) {

            // Calculate the Jacobian matrix.

            iflag[1] = 2;

            Minpack_f77.fdjac2_f77(nlls, m, n, x, fvec, fjac, iflag, epsfcn, wa4);

            nfev[1] += n;

            if (iflag[1] < 0) {

                // Termination

                info[1] = iflag[1];
                iflag[1] = 0;

                if (nprint > 0) {

                    nlls.fcn(m, n, x, fvec, iflag);

                }

                return;

            }

            // If requested, call fcn to enable printing of iterates.

            if (nprint > 0) {

                iflag[1] = 0;

                if ((iter - 1) % nprint == 0) {

                    nlls.fcn(m, n, x, fvec, iflag);

                }

                if (iflag[1] < 0) {

                    // Termination

                    info[1] = iflag[1];
                    iflag[1] = 0;

                    nlls.fcn(m, n, x, fvec, iflag);

                    return;

                }

            }

            // Compute the qr factorization of the Jacobian.

            Minpack_f77.qrfac_f77(m, n, fjac, true, ipvt, wa1, wa2, wa3);

            // On the first iteration and if mode is 1, scale according
            // to the norms of the columns of the initial Jacobian.

            if (iter == 1) {

                if (mode != 2) {

                    for (j = 1; j <= n; j++) {

                        diag[j] = wa2[j];

                        if (wa2[j] == zero)
                            diag[j] = one;

                    }

                }

                // On the first iteration, calculate the norm of the scaled x
                // and initialize the step bound delta.

                for (j = 1; j <= n; j++) {

                    wa3[j] = diag[j] * x[j];

                }

                xnorm = Minpack_f77.enorm_f77(n, wa3);

                delta = factor * xnorm;

                if (delta == zero)
                    delta = factor;

            }

            // Form (q transpose)*fvec and store the first n components in
            // qtf.

            for (i = 1; i <= m; i++)
                wa4[i] = fvec[i];

            for (j = 1; j <= n; j++) {

                if (fjac[j][j] != zero) {

                    sum = zero;

                    for (i = j; i <= m; i++)
                        sum += fjac[i][j] * wa4[i];

                    temp = -sum / fjac[j][j];

                    for (i = j; i <= m; i++)
                        wa4[i] += fjac[i][j] * temp;

                }

                fjac[j][j] = wa1[j];
                qtf[j] = wa4[j];

            }

            // Compute the norm of the scaled gradient.

            gnorm = zero;

            if (fnorm != zero) {

                for (j = 1; j <= n; j++) {

                    l = ipvt[j];

                    if (wa2[l] != zero) {

                        sum = zero;

                        for (i = 1; i <= j; i++)
                            sum += fjac[i][j] * (qtf[i] / fnorm);

                        gnorm = Math.max(gnorm, Math.abs(sum / wa2[l]));

                    }

                }

            }

            // Test for convergence of the gradient norm.

            if (gnorm <= gtol)
                info[1] = 4;

            if (info[1] != 0) {

                // Termination

                if (iflag[1] < 0)
                    info[1] = iflag[1];
                iflag[1] = 0;

                if (nprint > 0) {

                    nlls.fcn(m, n, x, fvec, iflag);

                }

                return;

            }

            // Rescale if necessary.

            if (mode != 2) {

                for (j = 1; j <= n; j++) {

                    diag[j] = Math.max(diag[j], wa2[j]);

                }

            }

            // Beginning of the inner loop.

            donein = false;

            while (!donein) {

                // Determine the Levenberg-Marquardt parameter.

                Minpack_f77.lmpar_f77(n, fjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4);

                // Store the direction p and x + p.  Calculate the norm of p.

                for (j = 1; j <= n; j++) {

                    wa1[j] = -wa1[j];
                    wa2[j] = x[j] + wa1[j];
                    wa3[j] = diag[j] * wa1[j];

                }

                pnorm = Minpack_f77.enorm_f77(n, wa3);

                // On the first iteration, adjust the initial step bound.

                if (iter == 1)
                    delta = Math.min(delta, pnorm);

                // Evaluate the function at x + p and calculate its norm.

                iflag[1] = 1;

                nlls.fcn(m, n, wa2, wa4, iflag);

                nfev[1]++;

                if (iflag[1] < 0) {

                    // Termination

                    info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, iflag);

                    }

                    return;

                }

                fnorm1 = Minpack_f77.enorm_f77(m, wa4);

                // Compute the scaled actual reduction.

                actred = -one;

                if (p1 * fnorm1 < fnorm)
                    actred = one - (fnorm1 / fnorm) * (fnorm1 / fnorm);

                // Compute the scaled predicted reduction and
                // the scaled directional derivative.

                for (j = 1; j <= n; j++) {

                    wa3[j] = zero;
                    l = ipvt[j];
                    temp = wa1[l];

                    for (i = 1; i <= j; i++)
                        wa3[i] += fjac[i][j] * temp;

                }

                temp1 = Minpack_f77.enorm_f77(n, wa3) / fnorm;
                temp2 = (Math.sqrt(par[1]) * pnorm) / fnorm;

                prered = temp1 * temp1 + temp2 * temp2 / p5;
                dirder = -(temp1 * temp1 + temp2 * temp2);

                // Compute the ratio of the actual to the predicted
                // reduction.

                ratio = zero;
                if (prered != zero)
                    ratio = actred / prered;

                // Update the step bound.

                if (ratio <= p25) {

                    if (actred >= zero) {

                        temp = p5;

                    } else {

                        temp = p5 * dirder / (dirder + p5 * actred);

                    }

                    if (p1 * fnorm1 >= fnorm || temp < p1)
                        temp = p1;

                    delta = temp * Math.min(delta, pnorm / p1);

                    par[1] /= temp;

                } else {

                    if (par[1] == zero || ratio >= p75) {

                        delta = pnorm / p5;
                        par[1] *= p5;

                    }

                }

                // Test for successful iteration.

                if (ratio >= p0001) {

                    // Successful iteration.  Update x, fvec, and their norms.

                    for (j = 1; j <= n; j++) {

                        x[j] = wa2[j];
                        wa2[j] = diag[j] * x[j];

                    }

                    for (i = 1; i <= m; i++)
                        fvec[i] = wa4[i];

                    xnorm = Minpack_f77.enorm_f77(n, wa2);

                    fnorm = fnorm1;

                    iter++;

                }

                // Tests for convergence.

                if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one)
                    info[1] = 1;

                if (delta <= xtol * xnorm)
                    info[1] = 2;

                if (Math.abs(actred) <= ftol && prered <= ftol && p5 * ratio <= one && info[1] == 2)
                    info[1] = 3;

                if (info[1] != 0) {

                    // Termination

                    if (iflag[1] < 0)
                        info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, iflag);

                    }

                    return;

                }

                // Tests for termination and stringent tolerances.

                if (nfev[1] >= maxfev)
                    info[1] = 5;

                if (Math.abs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= one)
                    info[1] = 6;

                if (delta <= epsmch * xnorm)
                    info[1] = 7;

                if (gnorm <= epsmch)
                    info[1] = 8;

                if (info[1] != 0) {

                    // Termination

                    if (iflag[1] < 0)
                        info[1] = iflag[1];
                    iflag[1] = 0;

                    if (nprint > 0) {

                        nlls.fcn(m, n, x, fvec, iflag);

                    }

                    return;

                }

                // End of the inner loop.  Repeat if iteration unsuccessful.

                if (ratio >= p0001)
                    donein = true;

            }

            // End of the outer loop.

        }

    }

    /**
     * 
     *<p>
     * The fdjac2 method computes a forward-difference approximation to the m by
     * n Jacobian matrix associated with a specified problem of m functions in n
     * variables.
     *<p>
     * Translated by Steve Verrill on November 24, 2000 from the FORTRAN MINPACK
     * source produced by Garbow, Hillstrom, and More.
     * <p>
     * 
     * 
     *@param nlls
     *            A class that implements the Lmdif_fcn interface (see the
     *            definition in Lmdif_fcn.java). See LmdifTest_f77.java for an
     *            example of such a class. The class must define a method, fcn,
     *            that must have the form
     * 
     *            public static void fcn(int m, int n, double x[], double
     *            fvec[], int iflag[])
     * 
     *            The value of iflag[1] should not be changed by fcn unless the
     *            user wants to terminate execution of fdjac2_f77. In this case
     *            iflag[1] should be set to a negative integer.
     *@param m
     *            A positive integer set to the number of functions [number of
     *            observations]
     *@param n
     *            A positive integer set to the number of variables [number of
     *            parameters]. n must not exceed m.
     *@param x
     *            An input array.
     *@param fvec
     *            An input array that contains the functions evaluated at x.
     *@param fjac
     *            An output m by n array that contains the approximation to the
     *            Jacobian matrix evaluated at x.
     *@param iflag
     *            An integer variable that can be used to terminate the
     *            execution of fdjac2. See the description of nlls.
     *@param epsfcn
     *            An input variable used in determining a suitable step length
     *            for the forward-difference approximation. This approximation
     *            assumes that the relative errors in the functions are of the
     *            order of epsfcn. If epsfcn is less than the machine precision,
     *            it is assumed that the relative errors in the functions are of
     *            the order of the machine precision.
     *@param wa
     *            A work array.
     * 
     */

    public static void fdjac2_f77(Lmdif_fcn nlls, int m, int n, double x[], double fvec[], double fjac[][],
            int iflag[], double epsfcn, double wa[]) {

        /*

        Here is a copy of the fdjac2 FORTRAN documentation:


              subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
              integer m,n,ldfjac,iflag
              double precision epsfcn
              double precision x(n),fvec(m),fjac(ldfjac,n),wa(m)
        c     **********
        c
        c     subroutine fdjac2
        c
        c     this subroutine computes a forward-difference approximation
        c     to the m by n jacobian matrix associated with a specified
        c     problem of m functions in n variables.
        c
        c     the subroutine statement is
        c
        c       subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
        c
        c     where
        c
        c       fcn is the name of the user-supplied subroutine which
        c         calculates the functions. fcn must be declared
        c         in an external statement in the user calling
        c         program, and should be written as follows.
        c
        c         subroutine fcn(m,n,x,fvec,iflag)
        c         integer m,n,iflag
        c         double precision x(n),fvec(m)
        c         ----------
        c         calculate the functions at x and
        c         return this vector in fvec.
        c         ----------
        c         return
        c         end
        c
        c         the value of iflag should not be changed by fcn unless
        c         the user wants to terminate execution of fdjac2.
        c         in this case set iflag to a negative integer.
        c
        c       m is a positive integer input variable set to the number
        c         of functions.
        c
        c       n is a positive integer input variable set to the number
        c         of variables. n must not exceed m.
        c
        c       x is an input array of length n.
        c
        c       fvec is an input array of length m which must contain the
        c         functions evaluated at x.
        c
        c       fjac is an output m by n array which contains the
        c         approximation to the jacobian matrix evaluated at x.
        c
        c       ldfjac is a positive integer input variable not less than m
        c         which specifies the leading dimension of the array fjac.
        c
        c       iflag is an integer variable which can be used to terminate
        c         the execution of fdjac2. see description of fcn.
        c
        c       epsfcn is an input variable used in determining a suitable
        c         step length for the forward-difference approximation. this
        c         approximation assumes that the relative errors in the
        c         functions are of the order of epsfcn. if epsfcn is less
        c         than the machine precision, it is assumed that the relative
        c         errors in the functions are of the order of the machine
        c         precision.
        c
        c       wa is a work array of length m.
        c
        c     subprograms called
        c
        c       user-supplied ...... fcn
        c
        c       minpack-supplied ... dpmpar
        c
        c       fortran-supplied ... dabs,dmax1,dsqrt
        c
        c     argonne national laboratory. minpack project. march 1980.
        c     burton s. garbow, kenneth e. hillstrom, jorge j. more
        c
        c     **********

        */

        int i, j;
        //      double eps,h,temp,zero;
        double eps, h, temp;

        //      zero = 0.0;

        eps = Math.sqrt(Math.max(epsfcn, epsmch));

        for (j = 1; j <= n; j++) {

            temp = x[j];
            h = eps * Math.abs(temp);

            if (h == zero)
                h = eps;

            x[j] = temp + h;

            nlls.fcn(m, n, x, wa, iflag);

            if (iflag[1] < 0) {

                return;

            }

            x[j] = temp;

            for (i = 1; i <= m; i++) {

                fjac[i][j] = (wa[i] - fvec[i]) / h;

            }

        }

        return;

    }

}
