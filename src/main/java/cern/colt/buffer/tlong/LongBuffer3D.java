/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.buffer.tlong;

import cern.colt.list.tlong.LongArrayList;

/**
 * Fixed sized (non resizable) streaming buffer connected to a target
 * <tt>LongBuffer3DConsumer</tt> to which data is automatically flushed upon
 * buffer overflow.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class LongBuffer3D extends cern.colt.PersistentObject implements LongBuffer3DConsumer {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected LongBuffer3DConsumer target;

    protected long[] xElements;

    protected long[] yElements;

    protected long[] zElements;

    // vars cached for speed
    protected LongArrayList xList;

    protected LongArrayList yList;

    protected LongArrayList zList;

    protected int capacity;

    protected int size;

    /**
     * Constructs and returns a new buffer with the given target.
     * 
     * @param target
     *            the target to flush to.
     * @param capacity
     *            the number of points the buffer shall be capable of holding
     *            before overflowing and flushing to the target.
     */
    public LongBuffer3D(LongBuffer3DConsumer target, int capacity) {
        this.target = target;
        this.capacity = capacity;
        this.xElements = new long[capacity];
        this.yElements = new long[capacity];
        this.zElements = new long[capacity];
        this.xList = new LongArrayList(xElements);
        this.yList = new LongArrayList(yElements);
        this.zList = new LongArrayList(zElements);
        this.size = 0;
    }

    /**
     * Adds the specified point (x,y,z) to the receiver.
     * 
     * @param x
     *            the x-coordinate of the point to add.
     * @param y
     *            the y-coordinate of the point to add.
     * @param z
     *            the z-coordinate of the point to add.
     */
    public void add(long x, long y, long z) {
        if (this.size == this.capacity)
            flush();
        this.xElements[this.size] = x;
        this.yElements[this.size] = y;
        this.zElements[this.size++] = z;
    }

    /**
     * Adds all specified (x,y,z) points to the receiver.
     * 
     * @param xElements
     *            the x-coordinates of the points.
     * @param yElements
     *            the y-coordinates of the points.
     * @param zElements
     *            the y-coordinates of the points.
     */
    public void addAllOf(LongArrayList xElements, LongArrayList yElements, LongArrayList zElements) {
        int listSize = xElements.size();
        if (this.size + listSize >= this.capacity)
            flush();
        this.target.addAllOf(xElements, yElements, zElements);
    }

    /**
     * Sets the receiver's size to zero. In other words, forgets about any
     * internally buffered elements.
     */
    public void clear() {
        this.size = 0;
    }

    /**
     * Adds all internally buffered points to the receiver's target, then resets
     * the current buffer size to zero.
     */
    public void flush() {
        if (this.size > 0) {
            xList.setSize(this.size);
            yList.setSize(this.size);
            zList.setSize(this.size);
            this.target.addAllOf(xList, yList, zList);
            this.size = 0;
        }
    }
}
