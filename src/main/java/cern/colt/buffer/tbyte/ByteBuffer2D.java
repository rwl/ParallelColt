/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.buffer.tbyte;

import cern.colt.list.tbyte.ByteArrayList;

/**
 * Fixed sized (non resizable) streaming buffer connected to a target
 * <tt>ByteBuffer2DConsumer</tt> to which data is automatically flushed upon
 * buffer overflow.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class ByteBuffer2D extends cern.colt.PersistentObject implements ByteBuffer2DConsumer {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected ByteBuffer2DConsumer target;

    protected byte[] xElements;

    protected byte[] yElements;

    // vars cached for speed
    protected ByteArrayList xList;

    protected ByteArrayList yList;

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
    public ByteBuffer2D(ByteBuffer2DConsumer target, int capacity) {
        this.target = target;
        this.capacity = capacity;
        this.xElements = new byte[capacity];
        this.yElements = new byte[capacity];
        this.xList = new ByteArrayList(xElements);
        this.yList = new ByteArrayList(yElements);
        this.size = 0;
    }

    /**
     * Adds the specified point (x,y) to the receiver.
     * 
     * @param x
     *            the x-coordinate of the point to add.
     * @param y
     *            the y-coordinate of the point to add.
     */
    public void add(byte x, byte y) {
        if (this.size == this.capacity)
            flush();
        this.xElements[this.size] = x;
        this.yElements[this.size++] = y;
    }

    /**
     * Adds all specified points (x,y) to the receiver.
     * 
     * @param x
     *            the x-coordinates of the points to add.
     * @param y
     *            the y-coordinates of the points to add.
     */
    public void addAllOf(ByteArrayList x, ByteArrayList y) {
        int listSize = x.size();
        if (this.size + listSize >= this.capacity)
            flush();
        this.target.addAllOf(x, y);
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
            this.target.addAllOf(xList, yList);
            this.size = 0;
        }
    }
}
