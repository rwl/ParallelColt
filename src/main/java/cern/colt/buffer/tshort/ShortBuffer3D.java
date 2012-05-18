/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.buffer.tshort;

import cern.colt.list.tshort.ShortArrayList;

/**
 * Fixed sized (non resizable) streaming buffer connected to a target
 * <tt>ShortBuffer3DConsumer</tt> to which data is automatically flushed upon
 * buffer overflow.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class ShortBuffer3D extends cern.colt.PersistentObject implements ShortBuffer3DConsumer {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected ShortBuffer3DConsumer target;

    protected short[] xElements;

    protected short[] yElements;

    protected short[] zElements;

    // vars cached for speed
    protected ShortArrayList xList;

    protected ShortArrayList yList;

    protected ShortArrayList zList;

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
    public ShortBuffer3D(ShortBuffer3DConsumer target, int capacity) {
        this.target = target;
        this.capacity = capacity;
        this.xElements = new short[capacity];
        this.yElements = new short[capacity];
        this.zElements = new short[capacity];
        this.xList = new ShortArrayList(xElements);
        this.yList = new ShortArrayList(yElements);
        this.zList = new ShortArrayList(zElements);
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
    public void add(short x, short y, short z) {
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
    public void addAllOf(ShortArrayList xElements, ShortArrayList yElements, ShortArrayList zElements) {
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
