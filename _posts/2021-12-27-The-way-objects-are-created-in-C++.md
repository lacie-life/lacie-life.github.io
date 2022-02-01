---
title: The way objects are created in C++
author:
  name: Life Zero
  link: https://github.com/lacie-life
date:  2021-12-27 11:11:11 +0700
categories: [C++]
tags: [tutorial]
render_with_liquid: false
---

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

# Returning Objects From Functions

If you are returning anything bigger than a word-sized value, there are several ways of returning something from a function. The first, and most obvious, is:

```
Foo make_foo(int n)
{
    return Foo{n};
}
```

It may appear to you that, using the preceding, a full copy of Foo is being made, thereby wasting valuable resources. But it isn’t always so. Say you define Foo as:

```
struct Foo
{
    Foo(int n) {}
    Foo(const Foo&) 
    {
        cout << "COPY CONSTRUCTOR!!!! \n";
    }
};
```

You will find that the copy constructor may be called anywhere from zero to two times: the exact number of calls depends on the compiler. Return Value Optimization (RVO) is a compiler feature that specifically prevents those extra copies being made (since they don’t really affect how the code behaves). In complex scenarios, however, you really cannot rely on RVO happening, but when it comes to choosing whether or not to optimize return values, I prefer to follow Donald Knuth (book: The Art of Computer Programming).

Another approach is, of course, to simply return a smart pointer such as a unique_ptr:

```
unique_ptr<Foo> make_foo(int n)
{
    return make_unique<Foo>(n);
}
```

This is very safe, but also opinionated: you’ve chosen the smart pointer for the user. What if they don’t like smart pointers? What if they would prefer a shared_ptr instead?

The third and final option is to use a raw pointer, perhaps in tandem with GSL’s owner<T>. This way, you are not enforcing the clean-up of the allocated object, but you are sending a very clear message that it is the caller’s responsibility:

```
owner<Foo*> make_foo(int)
{
    return new Foo(n);
}
```

You can consider this approach as giving the user a hint: I’m returning a pointer and it’s up to you to take care of the pointer from now on. Of course, now the caller of make_foo() needs to handle the pointer: either by correctly calling delete or by wrapping it in a unique_ptr or shared_ptr. Keep in mind that owner<T> says nothing about copying.
All of these options are equally valid, and it’s difficult to say which option is better.


