/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.algo;

import cern.colt.matrix.AbstractFormatter;
import cern.colt.matrix.AbstractMatrix1D;
import cern.colt.matrix.AbstractMatrix2D;
import cern.colt.matrix.Former;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import cern.colt.matrix.tint.IntMatrix3D;

/**
 * Flexible, well human readable matrix print formatting; By default decimal
 * point aligned. Currenly works on 1-d, 2-d and 3-d matrices. Note that in most
 * cases you will not need to get familiar with this class; just call
 * <tt>matrix.toString()</tt> and be happy with the default formatting. This
 * class is for advanced requirements.
 * 
 * <p>
 * <b>Examples:</b>
 * <p>
 * Examples demonstrate usage on 2-d matrices. 1-d and 3-d matrices formatting
 * works very similar.
 * <table border="1" cellspacing="0">
 * <tr align="center">
 * <td>Original matrix</td>
 * </tr>
 * <tr>
 * <td>
 * 
 * <p>
 * <tt>int[][] values = {<br>
 {3, 0, -3.4, 0},<br>
 {5.1 ,0, +3.0123456789, 0}, <br>
 {16.37, 0.0, 2.5, 0}, <br>
 {-16.3, 0, -3.012345678E-4, -1},<br>
 {1236.3456789, 0, 7, -1.2}<br>
 };<br>
 matrix = new DenseIntMatrix2D(values);</tt>
 * </p>
 * </td>
 * </tr>
 * </table>
 * <p>
 * &nbsp;
 * </p>
 * <table border="1" cellspacing="0">
 * <tr align="center">
 * <td><tt>format</tt></td>
 * <td valign="top"><tt>Formatter.toString(matrix);</tt></td>
 * <td valign="top"><tt>Formatter.toSourceCode(matrix);</tt></td>
 * </tr>
 * <tr>
 * <td><tt>%G </tt><br>
 * (default)</td>
 * <td align="left" valign="top"><tt>5&nbsp;x&nbsp;4&nbsp;matrix<br>
 &nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;3.012346&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;-0.000301&nbsp;-1&nbsp;&nbsp;<br>
 1236.345679&nbsp;0&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-1.2 
 </tt></td>
 * <td align="left" valign="top"><tt>{<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;&nbsp;3.012346,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;-0.000301,&nbsp;-1&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{1236.345679,&nbsp;0,&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;-1.2}<br>
 }; </tt></td>
 * </tr>
 * <tr>
 * <td><tt>%1.10G</tt></td>
 * <td align="left" valign="top"><tt>5&nbsp;x&nbsp;4&nbsp;matrix<br>
 &nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;3.0123456789&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;<br>
 &nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;-0.0003012346&nbsp;-1&nbsp;&nbsp;<br>
 1236.3456789&nbsp;0&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-1.2 
 </tt></td>
 * <td align="left" valign="top"><tt>{<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;&nbsp;3.0123456789,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0,&nbsp;-0.0003012346,&nbsp;-1&nbsp;&nbsp;},<br>
 &nbsp;&nbsp;&nbsp;{1236.3456789,&nbsp;0,&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;-1.2}<br>
 }; </tt></td>
 * </tr>
 * <tr>
 * <td><tt>%f</tt></td>
 * <td align="left" valign="top"> <tt> 5&nbsp;x&nbsp;4&nbsp;matrix<br>
 &nbsp;&nbsp;&nbsp;3.000000&nbsp;0.000000&nbsp;-3.400000&nbsp;&nbsp;0.000000<br>
 &nbsp;&nbsp;&nbsp;5.100000&nbsp;0.000000&nbsp;&nbsp;3.012346&nbsp;&nbsp;0.000000<br>
 &nbsp;&nbsp;16.370000&nbsp;0.000000&nbsp;&nbsp;2.500000&nbsp;&nbsp;0.000000<br>
 &nbsp;-16.300000&nbsp;0.000000&nbsp;-0.000301&nbsp;-1.000000<br>
 1236.345679&nbsp;0.000000&nbsp;&nbsp;7.000000&nbsp;-1.200000 </tt></td>
 * <td align="left" valign="top"><tt> {<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;3.000000,&nbsp;0.000000,&nbsp;-3.400000,&nbsp;&nbsp;0.000000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;5.100000,&nbsp;0.000000,&nbsp;&nbsp;3.012346,&nbsp;&nbsp;0.000000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;16.370000,&nbsp;0.000000,&nbsp;&nbsp;2.500000,&nbsp;&nbsp;0.000000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;-16.300000,&nbsp;0.000000,&nbsp;-0.000301,&nbsp;-1.000000},<br>
 &nbsp;&nbsp;&nbsp;{1236.345679,&nbsp;0.000000,&nbsp;&nbsp;7.000000,&nbsp;-1.200000}<br>
 }; </tt></td>
 * </tr>
 * <tr>
 * <td><tt>%1.2f</tt></td>
 * <td align="left" valign="top"><tt>5&nbsp;x&nbsp;4&nbsp;matrix<br>
 &nbsp;&nbsp;&nbsp;3.00&nbsp;0.00&nbsp;-3.40&nbsp;&nbsp;0.00<br>
 &nbsp;&nbsp;&nbsp;5.10&nbsp;0.00&nbsp;&nbsp;3.01&nbsp;&nbsp;0.00<br>
 &nbsp;&nbsp;16.37&nbsp;0.00&nbsp;&nbsp;2.50&nbsp;&nbsp;0.00<br>
 &nbsp;-16.30&nbsp;0.00&nbsp;-0.00&nbsp;-1.00<br>
 1236.35&nbsp;0.00&nbsp;&nbsp;7.00&nbsp;-1.20 </tt></td>
 * <td align="left" valign="top"><tt>{<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;3.00,&nbsp;0.00,&nbsp;-3.40,&nbsp;&nbsp;0.00},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;5.10,&nbsp;0.00,&nbsp;&nbsp;3.01,&nbsp;&nbsp;0.00},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;16.37,&nbsp;0.00,&nbsp;&nbsp;2.50,&nbsp;&nbsp;0.00},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;-16.30,&nbsp;0.00,&nbsp;-0.00,&nbsp;-1.00},<br>
 &nbsp;&nbsp;&nbsp;{1236.35,&nbsp;0.00,&nbsp;&nbsp;7.00,&nbsp;-1.20}<br>
 }; </tt></td>
 * </tr>
 * <tr>
 * <td><tt>%0.2e</tt></td>
 * <td align="left" valign="top"><tt>5&nbsp;x&nbsp;4&nbsp;matrix<br>
 &nbsp;3.00e+000&nbsp;0.00e+000&nbsp;-3.40e+000&nbsp;&nbsp;0.00e+000<br>
 &nbsp;5.10e+000&nbsp;0.00e+000&nbsp;&nbsp;3.01e+000&nbsp;&nbsp;0.00e+000<br>
 &nbsp;1.64e+001&nbsp;0.00e+000&nbsp;&nbsp;2.50e+000&nbsp;&nbsp;0.00e+000<br>
 -1.63e+001&nbsp;0.00e+000&nbsp;-3.01e-004&nbsp;-1.00e+000<br>
 &nbsp;1.24e+003&nbsp;0.00e+000&nbsp;&nbsp;7.00e+000&nbsp;-1.20e+000 </tt></td>
 * <td align="left" valign="top"><tt>{<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;3.00e+000,&nbsp;0.00e+000,&nbsp;-3.40e+000,&nbsp;&nbsp;0.00e+000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;5.10e+000,&nbsp;0.00e+000,&nbsp;&nbsp;3.01e+000,&nbsp;&nbsp;0.00e+000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;1.64e+001,&nbsp;0.00e+000,&nbsp;&nbsp;2.50e+000,&nbsp;&nbsp;0.00e+000},<br>
 &nbsp;&nbsp;&nbsp;{-1.63e+001,&nbsp;0.00e+000,&nbsp;-3.01e-004,&nbsp;-1.00e+000},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;1.24e+003,&nbsp;0.00e+000,&nbsp;&nbsp;7.00e+000,&nbsp;-1.20e+000}<br>
 }; </tt></td>
 * </tr>
 * <tr>
 * <td><tt>null</tt></td>
 * <td align="left" valign="top"><tt>5&nbsp;x&nbsp;4&nbsp;matrix <br>
 &nbsp;&nbsp;&nbsp;3.0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0<br>
 &nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0&nbsp;&nbsp;3.0123456789&nbsp;&nbsp;&nbsp;&nbsp;0.0<br>
 &nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0<br>
 &nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.0&nbsp;-3.012345678E-4&nbsp;-1.0<br>
 1236.3456789&nbsp;0.0&nbsp;&nbsp;7.0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-1.2 
 </tt> <tt> </tt></td>
 * <td align="left" valign="top"><tt> {<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;3.0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0.0,&nbsp;-3.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0.0},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0.0,&nbsp;&nbsp;3.0123456789&nbsp;&nbsp;,&nbsp;&nbsp;0.0},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;&nbsp;16.37&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0.0,&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;&nbsp;0.0},<br>
 &nbsp;&nbsp;&nbsp;{&nbsp;-16.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;0.0,&nbsp;-3.012345678E-4,&nbsp;-1.0},<br>
 &nbsp;&nbsp;&nbsp;{1236.3456789,&nbsp;0.0,&nbsp;&nbsp;7.0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;,&nbsp;-1.2}<br>
 }; </tt></td>
 * </tr>
 * </table>
 * 
 * <p>
 * Here are some more elaborate examples, adding labels for axes, rows, columns,
 * title and some statistical aggregations.
 * </p>
 * <table border="1" cellspacing="0">
 * <tr>
 * <td nowrap>
 * <p>
 * <tt> int[][] values = {<br>
 {5 ,10, 20, 40 },<br>
 { 7, 8 , 6 , 7 },<br>
 {12 ,10, 20, 19 },<br>
 { 3, 1 , 5 , 6 }<br>
 }; <br>
 </tt><tt>String title = "CPU performance over time [nops/sec]";<br>
 String columnAxisName = "Year";<br>
 String rowAxisName = "CPU"; <br>
 String[] columnNames = {"1996", "1997", "1998", "1999"};<br>
 String[] rowNames = { "PowerBar", "Benzol", "Mercedes", "Sparcling"};<br>
 hep.aida.bin.BinFunctions1D F = hep.aida.bin.BinFunctions1D.functions; // alias<br>
 hep.aida.bin.BinFunction1D[] aggr = {F.mean, F.rms, F.quantile(0.25), F.median, F.quantile(0.75), F.stdDev, F.min, F.max};<br>
 String format = "%1.2G";<br>
 IntMatrix2D matrix = new DenseIntMatrix2D(values); <br>
 new Formatter(format).toTitleString(<br>
 &nbsp;&nbsp;&nbsp;matrix,rowNames,columnNames,rowAxisName,columnAxisName,title,aggr); </tt>
 * </p>
 * </td>
 * </tr>
 * <tr>
 * <td><tt>
 CPU&nbsp;performance&nbsp;over&nbsp;time&nbsp;[nops/sec]<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;Year<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;1996&nbsp;&nbsp;1997&nbsp;&nbsp;1998&nbsp;&nbsp;1999&nbsp;&nbsp;|&nbsp;Mean&nbsp;&nbsp;RMS&nbsp;&nbsp;&nbsp;25%&nbsp;Q.&nbsp;Median&nbsp;75%&nbsp;Q.&nbsp;StdDev&nbsp;Min&nbsp;Max<br>
 ---------------------------------------------------------------------------------------<br>
 C&nbsp;PowerBar&nbsp;&nbsp;|&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;&nbsp;40&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;18.75&nbsp;23.05&nbsp;&nbsp;8.75&nbsp;&nbsp;15&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;25&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.48&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;40&nbsp;<br>
 P&nbsp;Benzol&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.04&nbsp;&nbsp;6.75&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.25&nbsp;&nbsp;&nbsp;0.82&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;8&nbsp;<br>
 U&nbsp;Mercedes&nbsp;&nbsp;|&nbsp;12&nbsp;&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;&nbsp;19&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;15.25&nbsp;15.85&nbsp;11.5&nbsp;&nbsp;&nbsp;15.5&nbsp;&nbsp;&nbsp;19.25&nbsp;&nbsp;&nbsp;4.99&nbsp;&nbsp;10&nbsp;&nbsp;20&nbsp;<br>
 &nbsp;&nbsp;Sparcling&nbsp;|&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;3.75&nbsp;&nbsp;4.21&nbsp;&nbsp;2.5&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.25&nbsp;&nbsp;&nbsp;2.22&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;6&nbsp;<br>
 ---------------------------------------------------------------------------------------<br>
 &nbsp;&nbsp;Mean&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;6.75&nbsp;&nbsp;7.25&nbsp;12.75&nbsp;18&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;RMS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;7.53&nbsp;&nbsp;8.14&nbsp;14.67&nbsp;22.62&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;25%&nbsp;Q.&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;4.5&nbsp;&nbsp;&nbsp;6.25&nbsp;&nbsp;5.75&nbsp;&nbsp;6.75&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;Median&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9&nbsp;&nbsp;&nbsp;&nbsp;13&nbsp;&nbsp;&nbsp;&nbsp;13&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;75%&nbsp;Q.&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;8.25&nbsp;10&nbsp;&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;&nbsp;24.25&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;StdDev&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;3.86&nbsp;&nbsp;4.27&nbsp;&nbsp;8.38&nbsp;15.81&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;Min&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
 &nbsp;&nbsp;Max&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;12&nbsp;&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;&nbsp;19&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 </tt></td>
 * </tr>
 * <tr>
 * <td nowrap><tt> same as above, but now without aggregations<br>
 aggr=null; </tt></td>
 * </tr>
 * <tr>
 * <td><tt> CPU&nbsp;performance&nbsp;over&nbsp;time&nbsp;[nops/sec]<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;Year<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;1996&nbsp;1997&nbsp;1998&nbsp;1999<br>
 ---------------------------------<br>
 C&nbsp;PowerBar&nbsp;&nbsp;|&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;40&nbsp;&nbsp;<br>
 P&nbsp;Benzol&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;8&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;<br>
 U&nbsp;Mercedes&nbsp;&nbsp;|&nbsp;12&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;19&nbsp;&nbsp;<br>
 &nbsp;&nbsp;Sparcling&nbsp;|&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp; 
 </tt></td>
 * </tr>
 * <tr>
 * <td nowrap>
 * <p>
 * <tt> same as above, but now without rows labeled<br>
 aggr=null;<br>
 rowNames=null;<br>
 rowAxisName=null; </tt>
 * </p>
 * </td>
 * </tr>
 * <tr>
 * <td><tt>
 CPU&nbsp;performance&nbsp;over&nbsp;time&nbsp;[nops/sec]<br>
 Year<br>
 1996&nbsp;1997&nbsp;1998&nbsp;1999<br>
 -------------------<br>
 &nbsp;5&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;40&nbsp;&nbsp;<br>
 &nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;8&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;<br>
 12&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;20&nbsp;&nbsp;&nbsp;19&nbsp;&nbsp;<br>
 &nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;
 </tt></td>
 * </tr>
 * </table>
 * 
 * <p>
 * A column can be broader than specified by the parameter
 * <tt>minColumnWidth</tt> (because a cell may not fit into that width) but a
 * column is never smaller than <tt>minColumnWidth</tt>. Normally one does not
 * need to specify <tt>minColumnWidth</tt> (default is <tt>1</tt>). This
 * parameter is only interesting when wanting to print two distinct matrices
 * such that both matrices have the same column width, for example, to make it
 * easier to see which column of matrix A corresponds to which column of matrix
 * B.
 * </p>
 * 
 * <p>
 * <b>Implementation:</b>
 * </p>
 * 
 * <p>
 * Note that this class is by no means ment to be used for high performance I/O
 * (serialization is much quicker). It is ment to produce well human readable
 * output.
 * </p>
 * <p>
 * Analyzes the entire matrix before producing output. Each cell is converted to
 * a String as indicated by the given C-like format string. If <tt>null</tt> is
 * passed as format string, {@link java.lang.Int#toString(int)} is used instead,
 * yielding full precision.
 * </p>
 * <p>
 * Next, leading and trailing whitespaces are removed. For each column the
 * maximum number of characters before and after the decimal point is
 * determined. (No problem if decimal points are missing). Each cell is then
 * padded with leading and trailing blanks, as necessary to achieve decimal
 * point aligned, left justified formatting.
 * </p>
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.2, 11/30/99
 */
public class IntFormatter extends AbstractFormatter {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * Constructs and returns a matrix formatter with format <tt>"%G"</tt>.
     */
    public IntFormatter() {
        this("%d");
    }

    /**
     * Constructs and returns a matrix formatter.
     * 
     * @param format
     *            the given format used to convert a single cell value.
     */
    public IntFormatter(String format) {
        setFormat(format);
        setAlignment(DECIMAL);
    }

    /**
     * Converts a given cell to a String; no alignment considered.
     */
    protected String form(IntMatrix1D matrix, int index, Former formatter) {
        return formatter.form(matrix.get(index));
    }

    /**
     * Converts a given cell to a String; no alignment considered.
     */

    protected String form(AbstractMatrix1D matrix, int index, Former formatter) {
        return this.form((IntMatrix1D) matrix, index, formatter);
    }

    /**
     * Returns a string representations of all cells; no alignment considered.
     */
    public String[][] format(IntMatrix2D matrix) {
        String[][] strings = new String[matrix.rows()][matrix.columns()];
        for (int row = matrix.rows(); --row >= 0;)
            strings[row] = formatRow(matrix.viewRow(row));
        return strings;
    }

    /**
     * Returns a string representations of all cells; no alignment considered.
     */

    protected String[][] format(AbstractMatrix2D matrix) {
        return this.format((IntMatrix2D) matrix);
    }

    /**
     * Returns the index of the decimal point.
     */
    protected int indexOfDecimalPoint(String s) {
        int i = s.lastIndexOf('.');
        if (i < 0)
            i = s.lastIndexOf('e');
        if (i < 0)
            i = s.lastIndexOf('E');
        if (i < 0)
            i = s.length();
        return i;
    }

    /**
     * Returns the number of characters before the decimal point.
     */

    protected int lead(String s) {
        if (alignment.equals(DECIMAL))
            return indexOfDecimalPoint(s);
        return super.lead(s);
    }

    /**
     * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal
     * Java statement.
     * 
     * @param matrix
     *            the matrix to format.
     */
    public String toSourceCode(IntMatrix1D matrix) {
        IntFormatter copy = (IntFormatter) this.clone();
        copy.setPrintShape(false);
        copy.setColumnSeparator(", ");
        String lead = "{";
        String trail = "};";
        return lead + copy.toString(matrix) + trail;
    }

    /**
     * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal
     * Java statement.
     * 
     * @param matrix
     *            the matrix to format.
     */
    public String toSourceCode(IntMatrix2D matrix) {
        IntFormatter copy = (IntFormatter) this.clone();
        String b3 = blanks(3);
        copy.setPrintShape(false);
        copy.setColumnSeparator(", ");
        copy.setRowSeparator("},\n" + b3 + "{");
        String lead = "{\n" + b3 + "{";
        String trail = "}\n};";
        return lead + copy.toString(matrix) + trail;
    }

    /**
     * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal
     * Java statement.
     * 
     * @param matrix
     *            the matrix to format.
     */
    public String toSourceCode(IntMatrix3D matrix) {
        IntFormatter copy = (IntFormatter) this.clone();
        String b3 = blanks(3);
        String b6 = blanks(6);
        copy.setPrintShape(false);
        copy.setColumnSeparator(", ");
        copy.setRowSeparator("},\n" + b6 + "{");
        copy.setSliceSeparator("}\n" + b3 + "},\n" + b3 + "{\n" + b6 + "{");
        String lead = "{\n" + b3 + "{\n" + b6 + "{";
        String trail = "}\n" + b3 + "}\n}";
        return lead + copy.toString(matrix) + trail;
    }

    /**
     * Returns a string representation of the given matrix.
     * 
     * @param matrix
     *            the matrix to convert.
     */
    public String toString(IntMatrix1D matrix) {
        IntMatrix2D easy = matrix.like2D(1, (int) matrix.size());
        easy.viewRow(0).assign(matrix);
        return toString(easy);
    }

    /**
     * Returns a string representation of the given matrix.
     * 
     * @param matrix
     *            the matrix to convert.
     */
    public String toString(IntMatrix2D matrix) {
        return super.toString(matrix);
    }

    /**
     * Returns a string representation of the given matrix.
     * 
     * @param matrix
     *            the matrix to convert.
     */
    public String toString(IntMatrix3D matrix) {
        StringBuffer buf = new StringBuffer();
        boolean oldPrintShape = this.printShape;
        this.printShape = false;
        for (int slice = 0; slice < matrix.slices(); slice++) {
            if (slice != 0)
                buf.append(sliceSeparator);
            buf.append(toString(matrix.viewSlice(slice)));
        }
        this.printShape = oldPrintShape;
        if (printShape)
            buf.insert(0, shape(matrix) + "\n");
        return buf.toString();
    }

    /**
     * Returns a string representation of the given matrix.
     * 
     * @param matrix
     *            the matrix to convert.
     */

    protected String toString(AbstractMatrix2D matrix) {
        return this.toString((IntMatrix2D) matrix);
    }

    /**
     * Returns a string representation of the given matrix with axis as well as
     * rows and columns labeled. Pass <tt>null</tt> to one or more parameters to
     * indicate that the corresponding decoration element shall not appear in
     * the string converted matrix.
     * 
     * @param matrix
     *            The matrix to format.
     * @param rowNames
     *            The headers of all rows (to be put to the left of the matrix).
     * @param columnNames
     *            The headers of all columns (to be put to above the matrix).
     * @param rowAxisName
     *            The label of the y-axis.
     * @param columnAxisName
     *            The label of the x-axis.
     * @param title
     *            The overall title of the matrix to be formatted.
     * @return the matrix converted to a string.
     */
    protected String toTitleString(IntMatrix2D matrix, String[] rowNames, String[] columnNames, String rowAxisName,
            String columnAxisName, String title) {
        if (matrix.size() == 0)
            return "Empty matrix";
        String[][] s = format(matrix);
        // String oldAlignment = this.alignment;
        // this.alignment = DECIMAL;
        align(s);
        // this.alignment = oldAlignment;
        return new cern.colt.matrix.tobject.algo.ObjectFormatter().toTitleString(
                cern.colt.matrix.tobject.ObjectFactory2D.dense.make(s), rowNames, columnNames, rowAxisName,
                columnAxisName, title);
    }

    /**
     * Returns a string representation of the given matrix with axis as well as
     * rows and columns labeled. Pass <tt>null</tt> to one or more parameters to
     * indicate that the corresponding decoration element shall not appear in
     * the string converted matrix.
     * 
     * @param matrix
     *            The matrix to format.
     * @param sliceNames
     *            The headers of all slices (to be put above each slice).
     * @param rowNames
     *            The headers of all rows (to be put to the left of the matrix).
     * @param columnNames
     *            The headers of all columns (to be put to above the matrix).
     * @param sliceAxisName
     *            The label of the z-axis (to be put above each slice).
     * @param rowAxisName
     *            The label of the y-axis.
     * @param columnAxisName
     *            The label of the x-axis.
     * @param title
     *            The overall title of the matrix to be formatted.
     * @return the matrix converted to a string.
     */
    private String xtoTitleString(IntMatrix3D matrix, String[] sliceNames, String[] rowNames, String[] columnNames,
            String sliceAxisName, String rowAxisName, String columnAxisName, String title) {
        if (matrix.size() == 0)
            return "Empty matrix";
        StringBuffer buf = new StringBuffer();
        for (int i = 0; i < matrix.slices(); i++) {
            if (i != 0)
                buf.append(sliceSeparator);
            buf.append(toTitleString(matrix.viewSlice(i), rowNames, columnNames, rowAxisName, columnAxisName, title
                    + "\n" + sliceAxisName + "=" + sliceNames[i]));
        }
        return buf.toString();
    }
}
