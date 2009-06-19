/* ***** BEGIN LICENSE BLOCK *****
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2008
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jplasma.tdouble;

/**
 * User's API.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dplasma {

    private Dplasma() {
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  PLASMA constants - CBLAS & LAPACK
     */
    public static final int PlasmaNoTrans = 111;
    public static final int PlasmaTrans = 112;
    public static final int PlasmaConjTrans = 113;

    public static final int PlasmaUpper = 121;
    public static final int PlasmaLower = 122;

    public static final int PlasmaNonUnit = 131;
    public static final int PlasmaUnit = 132;

    public static final int PlasmaLeft = 141;
    public static final int PlasmaRight = 142;

    public static final int PlasmaForward = 391;
    public static final int PlasmaBackward = 392;

    public static final int PlasmaColumnwise = 401;
    public static final int PlasmaRowwise = 402;

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  PLASMA constants - success & error codes
     */
    public static final int PLASMA_SUCCESS = 0;
    public static final int PLASMA_ERR_NOT_INITIALIZED = -1;
    public static final int PLASMA_ERR_REINITIALIZED = -2;
    public static final int PLASMA_ERR_NOT_SUPPORTED = -3;
    public static final int PLASMA_ERR_ILLEGAL_VALUE = -4;
    public static final int PLASMA_ERR_NOT_FOUND = -5;
    public static final int PLASMA_ERR_OUT_OF_MEMORY = -6;
    public static final int PLASMA_ERR_INTERNAL_LIMIT = -7;
    public static final int PLASMA_ERR_UNALLOCATED = -8;
    public static final int PLASMA_ERR_FILESYSTEM = -9;

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  PLASMA constants - configuration parameters
     */
    public static final int PLASMA_CONCURRENCY = 1;

    /**
     * Returns LAPACK string constant that corresponds to PLASMA integer
     * constant.
     * 
     * @param plasma_const
     *            PLASMA constant.
     * @return LAPACK constant.
     */
    public static String lapack_const(int plasma_const) {
        return Dcommon.plasma_lapack_constants[plasma_const];
    }

    /**
     * 
     * Set PLASMA integer parameter
     * 
     * @param param
     *            PLASMA parameter
     * @param value
     *            the value of the parameter.
     * @return Success or error code.
     */
    public static int plasma_set_int(int param, int value) {
        return Dauxiliary.plasma_set_int(param, value);
    }

    /**
     * Get PLASMA integer parameter
     * 
     * @param param
     *            PLASMA parameter
     * @return the value of the parameter or the error code
     */
    public static int plasma_get_int(int param) {
        return Dauxiliary.plasma_get_int(param);
    }

    /**
     * PLASMA initialization. This routine checks internal hardware constraints
     * and arranges PLASMA internal structures to fit the hardware.
     * 
     * @param M
     *            The number of rows.
     * @param N
     *            The number of columns.
     * @param NRHS
     *            Number of right hand sides.
     * @return Success or error code.
     */
    public static int plasma_Init(int M, int N, int NRHS) {
        return Dinitialize.plasma_Init(M, N, NRHS);
    }

    /**
     * PLASMA completion. This routine ends the parallel environment by joining
     * and destroy- ing the threads. Also, it releases any internal memory
     * allocation needed during the execution.
     * 
     * @return Success or error code.
     */
    public static int plasma_Finalize() {
        return Dinitialize.plasma_Finalize();
    }

    /**
     * This routine allocates the memory needed for the triangular factor T used
     * in QR and LQ factorization.
     * 
     * @param M
     *            The number of rows.
     * @param N
     *            The number of columns.
     * @return User's storage for T.
     */
    public static double[] plasma_Allocate_T(int M, int N) {
        return Dallocate.plasma_Allocate_T(M, N);
    }

    /**
     * This routine allocates the memory needed for the lower factor L used in
     * LU factorization.
     * 
     * @param M
     *            The number of rows.
     * @param N
     *            The number of columns.
     * @return User's storage for L.
     */
    public static double[] plasma_Allocate_L(int M, int N) {
        return Dallocate.plasma_Allocate_L(M, N);
    }

    /**
     * This routine allocates the memory needed for the pivot array IPIV used in
     * LU factorization.
     * 
     * @param M
     *            The number of rows.
     * @param N
     *            The number of columns.
     * @return User's storage for IPIV.
     */
    public static int[] plasma_Allocate_IPIV(int M, int N) {
        return Dallocate.plasma_Allocate_IPIV(M, N);
    }

    /**
     * Computes the solution to a real system of linear equations A * X = B,
     * where A is an N-by-N symmetric positive definite matrix and X and B are
     * N-by-NRHS matrices.
     * 
     * The Cholesky decomposition is used to factor A as A = U**T* U, if uplo =
     * PlasmaUpper, or A = L * L**T, if uplo = PlasmaLower, where U is an upper
     * triangular matrix and L is a lower triangular matrix. The factored form
     * of A is then used to solve the system of equations A * X = B.
     * 
     * 
     * @param uplo
     *            = PlasmaUpper: upper triangle of A is stored; = PlasmaLower:
     *            lower triangle of A is stored.
     * 
     * @param N
     *            The number of linear equations, i.e., the order of the matrix
     *            A. N >= 0.
     * 
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * 
     * @param A
     *            An array of dimension LDA-by-N. On entry, the symmetric matrix
     *            A. If if uplo = PlasmaUpper, the leading N-by-N upper
     *            triangular part of A contains the upper triangular part of the
     *            matrix A, and the strictly lower triangular part of A is not
     *            referenced. If if uplo = PlasmaLower, the leading N-by-N lower
     *            triangular part of A contains the lower triangular part of the
     *            matrix A, and the strictly upper triangular part of A is not
     *            referenced.
     * 
     *            On exit, if return value = PLASMA_SUCCESS, the factor U or L
     *            from the Cholesky factorization A = U**T*U or A = L*L**T.
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,N).
     * 
     * @param B
     *            An array of dimension LDB-by-NRHS. On entry, the N-by-NRHS
     *            right hand side matrix B. On exit, if return value =
     *            PLASMA_SUCCESS, the N-by-NRHS solution matrix X.
     * 
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDB >= max(1,N).
     * 
     * @return Success or error code.
     */
    public static int plasma_DPOSV(int uplo, int N, int NRHS, double[] A, int A_offset, int LDA, double[] B,
            int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dposv.plasma_DPOSV(uplo, N, NRHS, A, A_offset, LDA, B, B_offset, LDB);
    }

    /**
     * Computes the Cholesky factorization of a real symmetric positive definite
     * matrix A.
     * 
     * The factorization has the form A = U**T * U, if UPLO = 'U', or A = L *
     * L**T, if UPLO = 'L', where U is an upper triangular matrix and L is lower
     * triangular.
     * 
     * 
     * @param uplo
     *            = PlasmaUpper: upper triangle of A is stored; = PlasmaLower:
     *            lower triangle of A is stored.
     * @param N
     *            The order of the matrix A. N >= 0.
     * 
     * @param A
     *            An array of dimension LDA-by-N. On entry, the symmetric matrix
     *            A. If uplo = PlasmaUpper, the leading N-by-N upper triangular
     *            part of A contains the upper triangular part of the matrix A,
     *            and the strictly lower triangular part of A is not referenced.
     *            If uplo = PlasmaLower, the leading N-by-N lower triangular
     *            part of A contains the lower triangular part of the matrix A,
     *            and the strictly upper triangular part of A is not referenced.
     * 
     *            On exit, if return value = PLASMA_SUCCESS, the factor U or L
     *            from the Cholesky factorization A = U**T*U or A = L*L**T.
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,N).
     * @return Success or error code.
     */
    public static int plasma_DPOTRF(int uplo, int N, double[] A, int A_offset, int LDA) {
        return edu.emory.mathcs.jplasma.tdouble.Dpotrf.plasma_DPOTRF(uplo, N, A, A_offset, LDA);
    }

    /**
     * Solves a system of linear equations A*X = B with a symmetric positive
     * definite matrix A using the Cholesky factorization A = U**T*U or A =
     * L*L**T computed by DPOTRF.
     * 
     * @param uplo
     *            = PlasmaUpper: upper triangle of A is stored; = PlasmaLower:
     *            lower triangle of A is stored.
     * @param N
     *            The order of the matrix A. N >= 0.
     * 
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * 
     * @param A
     *            An array of dimension LDA-by-N. The triangular factor U or L
     *            from the Cholesky factorization
     * 
     *            A = U**T*U or A = L*L**T, as computed by plasma_DPOTRF.
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,N).
     * @param B
     *            An array of dimension LDB-by-NRHS. On entry, the right hand
     *            side matrix B. On exit, the solution matrix X.
     * 
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDB >= max(1,N).
     * 
     * @return Success or error code.
     */
    public static int plasma_DPOTRS(int uplo, int N, int NRHS, double[] A, int A_offset, int LDA, double[] B,
            int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dpotrs.plasma_DPOTRS(uplo, N, NRHS, A, A_offset, LDA, B, B_offset, LDB);
    }

    /**
     * Solves one of the matrix equations
     * 
     * op( A )*X = alpha*B, or X*op( A ) = alpha*B,
     * 
     * where alpha is a scalar, X and B are M-by-N matrices, A is a unit, or
     * non-unit, upper or lower triangular matrix and op( A ) is one of
     * 
     * op( A ) = A or op( A ) = A'.
     * 
     * The matrix X is overwritten on B. This routine works only for
     * side=PlasmaLeft.
     * 
     * 
     * @param side
     *            Specifies whether op( A ) appears on the left or right of X as
     *            follows:
     * 
     *            side = PlasmaLeft op( A )*X = alpha*B.
     * 
     *            side = PlasmaRight X*op( A ) = alpha*B.
     * 
     * @param uplo
     *            Specifies whether the matrix A is an upper or lower triangular
     *            matrix as follows:
     * 
     *            uplo = PlasmaUpper A is an upper triangular matrix.
     * 
     *            uplo = PlasmaLower A is a lower triangular matrix.
     * 
     * @param transA
     *            Specifies the form of op( A ) to be used in the matrix
     *            multiplication as follows:
     * 
     *            transA = PlasmaNoTrans op( A ) = A.
     * 
     *            transA = PlasmaTrans op( A ) = A'.
     * 
     * @param diag
     * 
     *            Specifies whether or not A is unit triangular as follows:
     * 
     *            diag = PlasmaUnit A is assumed to be unit triangular.
     * 
     *            diag = PlasmaNonUnit A is not assumed to be unit triangular.
     * 
     * @param N
     *            The number of linear equations, i.e., the order of the matrix
     *            A. N >= 0.
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * @param A
     *            An array of dimension LDA-by-K, where K is M when side =
     *            PlasmaLeft and is N when side = PlasmaRight. Before entry with
     *            uplo = PlasmaUpper, the leading K by K upper triangular part
     *            of the array A must contain the upper triangular matrix and
     *            the strictly lower triangular part of A is not referenced.
     *            Before entry with uplo = PlasmaLower, the leading K by K lower
     *            triangular part of the array A must contain the lower
     *            triangular matrix and the strictly upper triangular part of A
     *            is not referenced. Note that when diag = PlasmaUnit, the
     *            diagonal elements of A are not referenced either, but are
     *            assumed to be unity.
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,N).
     * @param B
     *            An array of DIMENSION LDB-by-N. Before entry, the leading
     *            M-by-N part of the array B must contain the right-hand side
     *            matrix B, and on exit is overwritten by the solution matrix X.
     * 
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDA >= max(1,M).
     * @return Success or error code.
     */
    public static int plasma_DTRSM(int side, int uplo, int transA, int diag, int N, int NRHS, double[] A, int A_offset,
            int LDA, double[] B, int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dtrsm.plasma_DTRSM(side, uplo, transA, diag, N, NRHS, A, A_offset, LDA,
                B, B_offset, LDB);
    }

    /**
     * Solves overdetermined real linear systems involving an M-by-N matrix A
     * using a QR factorization of A. It is assumed that A has full rank. The
     * following options are provided: <br>
     * i. If M >= N: find the least squares solution of an overdetermined
     * system, i.e., solve the least squares problem minimize || B - A*X ||.
     * This routine works only for trans = PlasmaNoTrans.
     * 
     * @param trans
     *            = PlasmaNoTrans: the linear system involves A; = PlasmaTrans:
     *            the linear system involves A**T.
     * 
     * @param M
     *            The number of rows of the matrix A. M >= 0.
     * 
     * @param N
     *            The number of columns of the matrix A. N >= 0.
     * 
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrices B and X. NRHS >=0.
     * 
     * @param A
     *            An array of dimension LDA-by-N On entry, the M-by-N matrix A.
     *            On exit, A is overwritten by details of its QR factorization
     *            as returned by plasma_DGEQRF;
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The index of the first element in the array A.
     * @param T
     *            The triangular factors. This array has to be allocated by
     *            plasma_Allocate_T.
     * @param T_offset
     *            The index of the first element in the array T_offset.
     * @param B
     *            An array of dimension LDB-by-NRHS On entry, the matrix B of
     *            right hand side vectors, stored columnwise; B is M-by-NRHS if
     *            trans = PlasmaNoTrans, or N-by-NRHS if trans = PlasmaTrans. On
     *            exit, B is overwritten by the solution vectors, stored
     *            columnwise: if trans = PlasmaNoTrans and M >= N, rows 1 to N
     *            of B contain the least squares solution vectors; the residual
     *            sum of squares for the solution in each column is given by the
     *            sum of squares of elements N+1 to M in that column; if trans =
     *            PlasmaNoTrans and M < N, rows 1 to N of B contain the minimum
     *            norm solution vectors; if trans = PlasmaTrans and M >= N, rows
     *            1 to M of B contain the minimum norm solution vectors; if
     *            trans = PlasmaTrans and M < N, rows 1 to M of B contain the
     *            least squares solution vectors; the residual sum of squares
     *            for the solution in each column is given by the sum of squares
     *            of elements M+1 to N in that column.
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDA >= max(1,M).
     * @return Success or error code.
     */
    public static int plasma_DGELS(int trans, int M, int N, int NRHS, double[] A, int A_offset, int LDA, double[] T,
            int T_offset, double[] B, int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dgels.plasma_DGELS(trans, M, N, NRHS, A, A_offset, LDA, T, T_offset, B,
                B_offset, LDB);
    }

    /**
     * Computes a QR factorization of a real M-by-N matrix A: A = Q * R.
     * 
     * 
     * @param M
     *            The number of rows of the matrix A. M >= 0.
     * 
     * @param N
     *            The number of columns of the matrix A. N >= 0.
     * 
     * @param A
     *            An array of dimension LDA,-by-N. On entry, the M-by-N matrix
     *            A. On exit, the elements on and above the diagonal of the
     *            array
     * 
     *            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
     *            upper triangular if M >= N); the elements below the diagonal,
     *            with the array TAU, represent the orthogonal matrix Q as a
     *            product of min(M,N) elementary reflectors (see Further
     *            Details).
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,M).
     * @param T
     *            An array allocated by plasma_Allocate_T. The scalar factors of
     *            the elementary reflectors (see Further Details).
     * 
     * @param T_offset
     *            The index of the first element in the array T.
     * @return Success or error code.
     *         <p>
     *         <b>Further Details</b>
     *         <p>
     * 
     *         The matrix Q is represented as a product of elementary reflectors
     * 
     *         Q = H(1) H(2) . . . H(k), where k = min(M,N).
     * 
     *         Each H(i) has the form
     * 
     *         H(i) = I - tau * v * v'
     * 
     *         where tau is a real scalar, and v is a real vector with v(1:i-1)
     *         = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i), and
     *         tau in T(i).
     */
    public static int plasma_DGEQRF(int M, int N, double[] A, int A_offset, int LDA, double[] T, int T_offset) {
        return Dgeqrf.plasma_DGEQRF(M, N, A, A_offset, LDA, T, T_offset);
    }

    /**
     * Overwrites the general real M-by-N matrix B with
     * <p>
     * <table border=1>
     * <tr align="center">
     * <td></td>
     * <td>side = PlasmaLeft</td>
     * <td>side = PlasmaRight</td>
     * </tr>
     * <tr align="center">
     * <td>trans = PlasmaNoTrans</td>
     * <td>Q * B</td>
     * <td>B * Q</td>
     * </tr>
     * <tr align="center">
     * <td>trans = PlasmaTrans</td>
     * <td>Q**T * B</td>
     * <td>B * Q**T</td>
     * </tr>
     * </table>
     * </p>
     * where Q is a real orthogonal matrix defined as the product of k
     * elementary reflectors
     * 
     * Q = H(1) H(2) . . . H(k)
     * 
     * as returned by plasma_DGEQRF. Q is of order M if side = PlasmaLeft and of
     * order N if side = PlasmaRight. This routine works only for side =
     * PlasmaLeft.
     * 
     * 
     * @param side
     *            = PlasmaLeft: apply Q or Q**T from the Left; = PlasmaRight:
     *            apply Q or Q**T from the Right.
     * @param trans
     *            = PlasmaNoTrans: No transpose, apply Q; = PlasmaTrans:
     *            Transpose, apply Q**T.
     * @param M
     *            The number of rows of the matrix B. M >= 0.
     * @param NRHS
     *            The number of right hand sides. NRHS >= 0.
     * @param N
     *            The number of columns of the matrix B. N >= 0.
     * @param A
     *            An array of dimension LDA-by-K. The i-th column must contain
     *            the vector which defines the elementary reflector H(i), for i
     *            = 1,2,...,k, as returned by plasma_DGEQRF in the first k
     *            columns of its array argument A. A is modified by the routine
     *            but restored on exit.
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. If side = PlasmaLeft,
     *            LDA >= max(1,M); if side = PlasmaRight, LDA >= max(1,N).
     * @param T
     *            T(i) must contain the scalar factor of the elementary
     *            reflector H(i), as returned by plasma_DGEQRF.
     * @param T_offset
     *            The index of the first element in the array T.
     * @param B
     *            An array of dimension LDB-by-N. On entry, the M-by-N matrix B.
     *            On exit, B is overwritten by Q*B or Q**T*B or B*Q**T or B*Q.
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array C. LDB >= max(1,M).
     * 
     * @return Success or error code.
     */
    public static int plasma_DORMQR(int side, int trans, int M, int NRHS, int N, double[] A, int A_offset, int LDA,
            double[] T, int T_offset, double[] B, int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dormqr.plasma_DORMQR(side, trans, M, NRHS, N, A, A_offset, LDA, T,
                T_offset, B, B_offset, LDB);
    }

    /**
     * Computes the solution to a real system of linear equations A * X = B,
     * where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
     * 
     * The LU decomposition with partial pivoting and row interchanges is used
     * to factor A as A = P * L * U, where P is a permutation matrix, L is unit
     * lower triangular, and U is upper triangular. The factored form of A is
     * then used to solve the system of equations A * X = B.
     * 
     * @param N
     *            The number of linear equations, i.e., the order of the matrix
     *            A. N >= 0.
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * @param A
     *            An array of dimension LDA-by-N. On entry, the N-by-N
     *            coefficient matrix A. On exit, the factors L and U from the
     *            factorization A = P*L*U; the unit diagonal elements of L are
     *            not stored.
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,N).
     * @param L
     *            The lower factor L. This array has to be allocated by
     *            plasma_Allocate_L.
     * @param L_offset
     *            The index of the first element in the array L.
     * @param IPIV
     *            The pivot indices that define the permutation matrix P; row i
     *            of the matrix was interchanged with row IPIV(i). This array
     *            has to be allocated by plasma_Allocate_IPIV.
     * @param IPIV_offset
     *            The index of the first element in the array IPIV.
     * @param B
     *            An array of dimension LDB-by-NRHS. On entry, the N-by-NRHS
     *            matrix of right hand side matrix B. On exit, if return value =
     *            PLASMA_SUCCESS, the N-by-NRHS solution matrix X.
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDB >= max(1,N).
     * @return Success or error code.
     */
    public static int plasma_DGESV(int N, int NRHS, double[] A, int A_offset, int LDA, double[] L, int L_offset,
            int[] IPIV, int IPIV_offset, double[] B, int B_offset, int LDB) {
        return Dgesv.plasma_DGESV(N, NRHS, A, A_offset, LDA, L, L_offset, IPIV, IPIV_offset, B, B_offset, LDB);
    }

    /**
     * Computes an LU factorization of a general M-by-N matrix A using partial
     * pivoting with row interchanges.
     * 
     * The factorization has the form A = P * L * U where P is a permutation
     * matrix, L is lower triangular with unit diagonal elements (lower
     * trapezoidal if M > N), and U is upper triangular (upper trapezoidal if M
     * < N).
     * 
     * @param M
     *            The number of rows of the matrix A. M >= 0.
     * @param N
     *            The number of columns of the matrix A. N >= 0.
     * @param A
     *            An array of dimension LDA-by-N. On entry, the M-by-N matrix to
     *            be factored. On exit, the factors L and U from the
     *            factorization A = P*L*U; the unit diagonal elements of L are
     *            not stored.
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,M).
     * @param L
     *            The lower factor L. This array has to be allocated by
     *            plasma_Allocate_L.
     * @param L_offset
     *            The index of the first element in the array L.
     * @param IPIV
     *            The pivot indices; for 1 <= i <= min(M,N), row i of the matrix
     *            was interchanged with row IPIV(i). This array has to be
     *            allocated by plasma_Allocate_IPIV.
     * @param IPIV_offset
     *            The index of the first element in the array IPIV.
     * @return Success or error code.
     */
    public static int plasma_DGETRF(int M, int N, double[] A, int A_offset, int LDA, double[] L, int L_offset,
            int[] IPIV, int IPIV_offset) {
        return edu.emory.mathcs.jplasma.tdouble.Dgetrf.plasma_DGETRF(M, N, A, A_offset, LDA, L, L_offset, IPIV,
                IPIV_offset);
    }

    /**
     * Solves a system of linear equations A * X = B with a general M-by-N
     * matrix A and a M-by-NRHS B matrix using the LU factorization computed by
     * plasma_DGETRF.
     * 
     * @param M
     *            The number of rows of the matrix A. M >= 0.
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * 
     * @param N
     *            The number of columns of the matrix A. M >= 0.
     * @param A
     *            An array of dimension LDA-by-N. The factors L and U from the
     *            factorization A = P*L*U as computed by plasma_DGETRF.
     * 
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,M).
     * @param L
     *            The lower factor L computed by plasma_DGETRF.
     * @param L_offset
     *            The index of the first element in the array L.
     * @param IPIV
     *            The pivot indices from plasma_DGETRF; for 1<=i<=N, row i of
     *            the matrix was interchanged with row IPIV(i).
     * @param IPIV_offset
     *            The index of the first element in the array IPIV.
     * @param B
     *            An array of dimension LDB-by-NRHS. On entry, the right hand
     *            side matrix B. On exit, the solution matrix X.
     * 
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDB >= max(1,N).
     * @return Success or error code.
     */
    public static int plasma_DGETRS(int M, int NRHS, int N, double[] A, int A_offset, int LDA, double[] L,
            int L_offset, int[] IPIV, int IPIV_offset, double[] B, int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dgetrs.plasma_DGETRS(M, NRHS, N, A, A_offset, LDA, L, L_offset, IPIV,
                IPIV_offset, B, B_offset, LDB);
    }

    /**
     * Applies the factor L with the pivot IPIV from plasma_DGETRF to solve
     * P*L*X = B, where L is an M-by-N matrix and X and B are N-by-NRHS
     * matrices.
     * 
     * @param M
     *            The number of rows of the matrix A. M >= 0.
     * @param NRHS
     *            The number of right hand sides, i.e., the number of columns of
     *            the matrix B. NRHS >= 0.
     * @param N
     *            The number of columns of the matrix A. M >= 0.
     * @param A
     *            An array of dimension LDA-by-N. The factors L and U from the
     *            factorization A = P*L*U as computed by plasma_DGETRF.
     * @param A_offset
     *            The index of the first element in the array A.
     * @param LDA
     *            The leading dimension of the array A. LDA >= max(1,M).
     * @param L
     *            The lower factor L computed by plasma_DGETRF.
     * @param L_offset
     *            The index of the first element in the array L.
     * @param IPIV
     *            The pivot indices from plasma_DGETRF; for 1<=i<=N, row i of
     *            the matrix was interchanged with row IPIV(i).
     * @param IPIV_offset
     *            The index of the first element in the array IPIV.
     * @param B
     *            An array of dimension LDB-by-NRHS. On entry, the right hand
     *            side matrix B. On exit, the solution matrix X.
     * @param B_offset
     *            The index of the first element in the array B.
     * @param LDB
     *            The leading dimension of the array B. LDB >= max(1,N).
     * @return Success or error code.
     */
    public static int plasma_DTRSMPL(int M, int NRHS, int N, double[] A, int A_offset, int LDA, double[] L,
            int L_offset, int[] IPIV, int IPIV_offset, double[] B, int B_offset, int LDB) {
        return edu.emory.mathcs.jplasma.tdouble.Dtrsmpl.plasma_DTRSMPL(M, NRHS, N, A, A_offset, LDA, L, L_offset, IPIV,
                IPIV_offset, B, B_offset, LDB);
    }
}
