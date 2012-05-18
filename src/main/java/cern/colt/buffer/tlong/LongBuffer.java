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
 * <tt>LongBufferConsumer</tt> to which data is automatically flushed upon
 * buffer overflow.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class LongBuffer extends cern.colt.PersistentObject implements LongBufferConsumer {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    protected LongBufferConsumer target;

    protected long[] elements;

    // vars cached for speed
    protected LongArrayList list;

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
    public LongBuffer(LongBufferConsumer target, int capacity) {
        this.target = target;
        this.capacity = capacity;
        this.elements = new long[capacity];
        this.list = new LongArrayList(elements);
        this.size = 0;
    }

    /**
     * Adds the specified element to the receiver.
     * 
     * @param element
     *            the element to add.
     */
    public void add(long element) {
        if (this.size == this.capacity)
            flush();
        this.elements[size++] = element;
    }

    /**
     * Adds all elements of the specified list to the receiver.
     * 
     * @param list
     *            the list of which all elements shall be added.
     */
    public void addAllOf(LongArrayList list) {
        int listSize = list.size();
        if (this.size + listSize >= this.capacity)
            flush();
        this.target.addAllOf(list);
    }

    /**
     * Sets the receiver's size to zero. In other words, forgets about any
     * internally buffered elements.
     */
    public void clear() {
        this.size = 0;
    }

    /**
     * Adds all internally buffered elements to the receiver's target, then
     * resets the current buffer size to zero.
     */
    public void flush() {
        if (this.size > 0) {
            list.setSize(this.size);
            this.target.addAllOf(list);
            this.size = 0;
        }
    }
}
