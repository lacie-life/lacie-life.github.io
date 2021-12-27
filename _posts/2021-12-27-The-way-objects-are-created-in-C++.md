# The way objects are created in C++

## Stack allocation 

Creates an object that will be allocated on the stack. The object will be cleaned up automatically at the end of the scope (you can make an artificial scope anywhere with a pair of curly braces). The object will call the destructor at the very end of the scope provided you assign this object to a variable; if you don’t, the destructor will be called immediately.

## Heap allocation

Using a raw pointer puts the object on the heap (a.k.a. the free store). Foo* foo = new Foo; creates a new instance of Foo and leaves open the question of who is in charge of cleaning up the object. The [GSL](https://github.com/Microsoft/GSL) owner<T> tries to introduce some idea of “ownership” of a raw pointer but doesn’t involve any cleanup code—you still have to write it yourself.

## Unique pointer (unique_ptr)

Can take a heapallocated pointer and manage it so that it’s cleaned up automatically when there is no longer a reference to it. A unique pointer really is unique: you cannot make copies of it, and you cannot pass it into another function without losing control of the original.

## Share pointer (shared_ptr)

Takes a heap-allocated pointer and manages it, but allows the sharing of this pointer around in code. The owned pointer is only cleaned up when there are no components holding on to the pointer.

## Weak pointer (weak_ptr)

It is a smart but nonowning pointer, holding a weak reference to an object managed by a shared_ptr. You need to convert it to a shared_ptr in order to be able to actually access the referenced object. One of its uses is to break circular references of shared_ptrs.

