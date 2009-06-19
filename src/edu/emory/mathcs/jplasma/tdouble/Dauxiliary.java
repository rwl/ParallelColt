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
 * Auxiliary routines.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
class Dauxiliary {

    private Dauxiliary() {
    }

    protected static void plasma_warning(String func_name, String msg_text) {
        System.err.println("PLASMA warning: " + func_name + "(): " + msg_text);
    }

    protected static void plasma_error(String func_name, String msg_text) {
        System.err.println("PLASMA error: " + func_name + "(): " + msg_text);
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
    protected static int plasma_set_int(int param, int value) {
        if (!Dcommon.plasma_cntrl.initialized) {
            plasma_warning("plasma_set_int", "PLASMA not initialized");
            return Dplasma.PLASMA_ERR_NOT_INITIALIZED;
        }
        switch (param) {
        case Dplasma.PLASMA_CONCURRENCY:
            if (value <= 0 || value > Dcommon.plasma_cntrl.cores_max) {
                plasma_warning("plasma_set_int", "illegal parameter value");
                return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
            }
            Dcommon.plasma_cntrl.cores_num = value;
            break;
        default:
            plasma_error("plasma_set_int", "illegal parameter value");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        return Dplasma.PLASMA_SUCCESS;
    }

    /**
     * Get PLASMA integer parameter
     * 
     * @param param
     *            PLASMA parameter
     * @return the value of the parameter or the error code
     */
    protected static int plasma_get_int(int param) {
        if (!Dcommon.plasma_cntrl.initialized) {
            plasma_warning("plasma_get_int", "PLASMA not initialized");
            return Dplasma.PLASMA_ERR_NOT_INITIALIZED;
        }
        switch (param) {
        case Dplasma.PLASMA_CONCURRENCY:
            return Dcommon.plasma_cntrl.cores_num;
        default:
            plasma_error("plasma_get_int", "illegal parameter value");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Clear auxiliary structure progress
     */
    protected static void plasma_clear_aux_progress(int size, int value) {
        int i;
        int[] progress = Dcommon.plasma_aux.progress;
        for (i = 0; i < size; i++)
            progress[i] = value;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Tune block size NB and internal block size IB
     */
    protected static int plasma_tune(int func, int M, int N, int NRHS) {
        Dcommon.plasma_cntrl.NB = 160;
        Dcommon.plasma_cntrl.IB = 40;
        /* Calculate A, B tile size and round up to cache line size */
        int NBNBSIZE = Dcommon.plasma_cntrl.NB * Dcommon.plasma_cntrl.NB;
        Dcommon.plasma_cntrl.NBNBSIZE = NBNBSIZE;
        /* Calculate T tile size and round up to cache line size */
        int IBNBSIZE = Dcommon.plasma_cntrl.IB * Dcommon.plasma_cntrl.NB;
        Dcommon.plasma_cntrl.IBNBSIZE = IBNBSIZE;
        return Dplasma.PLASMA_SUCCESS;
    }
}
