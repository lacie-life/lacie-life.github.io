---
title: Object Oriented Programming Keywords
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2021-12-31 11:11:11 +0700
categories: [C++, OOP]
tags: [tutorial]
render_with_liquid: false
---

# Object Oriented Programming Keywords

*Object Oriented Programming languages are defined by the following key words: abstraction, encapsulation, inheritance, and polymorphism. An object is a container of data and functions that affect the data. In an OOP, a "child" object can "extend" another object (making it more specific) by inheriting from a "parent" object. Thus the child gets all the parents data and functionality "for free". The idea of an "interface" is also key to OOPs. An interface is the allowed interactions between the outside world (e.g., other programs) and the object itself.*

## Object

Objects are at heart very simple. They are a way to represent information about a "real world" idea and they contain ways to manipulate that information. All of this "code" is encapsulated in an object recipe (Class file).

1. They contain information about something in the program.
2. They provide actions (also called functions or methods) which manipulate that information. By combining all the data and actions that can apply to an "object" into one piece of code, we make our programs easier to write and maintain.

Often an "object" in a computer program tries to model (i.e., approximate) a real world entity. For example a "car" could be modeled in a computer by information about the size, number of doors, engine, color, etc. of a car. Additionally there would be little pieces of assocaited code (called methods) which manipulate the car, for example: start_car() would "simulate" turning the engine on, break() would simulate slowing the car down, etc.

The interface between the car object and the rest of the program (or world) would be the putlic methods which are allowed on the car. For example, turn wheel left, turn wheel right, step on gas. We don't know (nor do we care) how these function work, just so long as the car does know and responds appropriately to each function call.

### Objects vs Classes

Classes are to Objects as Recipes are to Cakes.

The class file contains the blue print (or recipe) of how to build an object and information about what defines an object.

The "object" is the computer's Run Time representation of the data describing the object and the methods to manipulate the object.

There is a strong correlation between these two items, but it is important to remember, objects exist when the program is being run; the class file is written by the programmer to allow the computer to use/construct new objects.

### Part of an object

1. The Package (or namespace) that the class belongs to. Packages are just a way to allow you to have a "Bank" class, and me to have a "Bank" class and thus distinguish which Bank is which.

2. Class Name and Inheritance information.

The name of the class should be a symbolic representation of what the class represents. The inheritance information describes what "parent" class the object comes from. For example, a car doesn't really stand alone. A car really IS A vehicle. Likewise a truck IS A vehicle. All information that would be the same for both cars and trucks (and others) really should be placed in a Vehicle class, from which both cars and trucks extend.

3. Data Fields (sometimes called Member Variables) representing facts about the object.

4. Methods representing actions on the object.

Some common methods are:

- Constructor
- Setters
- Getters
- toString
- create_display_list (for objects to be displayed graphically).

#### Fields or Member Variables (or Object Level Variables)

The goal of an object is to contain enough information to (in the program) describe (or approximate) the idea. For example, we might have a "square" object. A square can be defined by the X,Y position of it's center, and the length of its sides. This is enough information to "recreate" a square.

#### Member Variable vs. Object Level Variable vs Field

The variables which contain the information which define an object will be usually be referred to as member variables, but sometimes as data fields (or just fields), or as "object level" variables.

Member variables are accessible from any function in a class file. For example, in the car example, when we say turn right, the car code really needs to know lots of things, like are we moving, how fast, is the break on, etc etc etc. All of these pieces of information are stored internally to the car in member variables. Every function in a class has access to all member variables, and any change made to a member variable (e.g., changing the speed from 10 mph to 5 mph) would affect the future operation of any other function.

#### Public vs. Private vs. Protected

When creating objects (or just reading the code for objects created by others) you will often see the key words: public, private, and protected. Here is a short description of what they mean:

<b> Public </b>

Any variable (or function) that is tagged "public" can be used by any "outsider" to look at or modify the current state of an object.

Most data associated with an object should not be public. Only variables that are often changed by the outside world (with out affecting the rest of the object) should be declared public. For example, the speed of a car would most definitely be private not public, because another program should not be able to stop a car by tellin the car its speed is 0. The only way an "outsider" can change the speed of a car is to use the "break()" function associated with the car.

<b> Private </b>

Any variable (or function) that is tagged "private" can only be used by the internal code of the object. This prevents the outside user of the code from manipulating the object except through the well defined interface.

Often functions that are tagged "private" are referred to as "helper" functions, because they are usually used by other functions in the object to complete "sub-tasks".

<b> Protected </b>

For the most part, you can read protected as "private". The exception to this is when we use the Object Oriented technique of Inheritance. Inheritance is when one class file is a CHILD of another class file, thus "getting" all the code from the parent class for free.

Children Objects which "extend" Parent objects have full access (public) to any protected variable (or function) in the parent object.

#### Methods

Once the data has been defined for what it means to be an object, then the programmer needs ways to manipulate that data. These ways are called functions.

An object's functions have access not only to their parameters (as all functions do) but also to ALL Member Variable!

### The Constructor

The constructor (for an object) is a method that is called (FOR YOU by the computer) whenever you craete a new object. The purpose of the constructor is to initialize values for all the member variables associated with the object.

In previous computer languages, objects (then called structures) were allowed to be created "empty". Empty meant: "What ever the heck happened to be in memory at the time". The developers of those languages "knew" that programmers would always initialize these objects directly after creating them... This is a great example of something that was "known" but was in fact NOT TRUE. This often created very hard to find mistakes in programs where future code accessed undefined variables.

Learning from these program language design errors, the creaters of new languages, like Java, ActionScript, and C++, developed the Constructor. The constructor is a special function that can only be used once, never returns a value, and sole responsibility is to make sure every data field in the object is intialized to a reasonable value to start.

#### Key Ideas behind the above constructor:

1. "Public"

The public key word tells the class file (and other programmers) that this object can be created by anyone. Public means anyone. Alternatively in some cases, you will see the key word "Private" next to a function. Again, this means that only the object itself can use the function.

Think of this in the same way you think of your "door bell" and your "dishwasher". Everyone can use the door bell (its public to the world). Only you are allowed to use your dishwasher (but perhaps other people could use the clean plates that the dishwasher produces...).

2. Parameters: (int starting_x, int starting_y, int starting_size)

Parameters are symbolic names for the information that must be presented to the object when it is created. The information in these symbolic names, which is provided by the "caller" of the method, is used to transform a "generic" version of the object into a specific version of the object. For example, we might set the x,y of one square to 100,200, and of another square to 500,600. The parameters allow the creator of our Square object to give it a "personality"

3. this - (the key word)

You will often notice the use of the key word "this". "This" refers to the current object. "This" is a runtime (as well as compile time) concept. The use of "this" is not required, but is used by the programmer to show that he or she knows that the thing being referred to is an "object level function or data value).

Local variables (those used only within a function) are not tagged with this.

### Friend Class and Friend Function

#### Friend class

Friend Class A friend class can access private and protected members of other class in which it is declared as friend. It is sometimes useful to allow a particular class to access private members of other class. For example, a LinkedList class may be allowed to access private members of Node. 

#### Friend Function

Friend Function Like friend class, a friend function can be given a special grant to access private and protected members. A friend function can be: 

a) A member of another class 

b) A global function 

#### Important points about friend functions and classes

- Friends should be used only for limited purpose. too many functions or external classes are declared as friends of a class with protected or private data, it lessens the value of encapsulation of separate classes in object-oriented programming.
- Friendship is not mutual. If class A is a friend of B, then B doesn’t become a friend of A automatically.
- Friendship is not inherited 
- The concept of friends is not there in Java. 

#### Friend and Encapsulation

Some people believe that the idea of having friend classes violates the principle of encapsulation because it means that one class can get at the internals of another. One way to think about this, however, is that friend is simply part of a class's overall interface that it shows the world. Just like an elevator repairman has access to a different interface than an elevator rider, some classes or functions require expanded access to the internals of another class. Moreover, using friend allows a class to present a more restrictive interface to the outside world by hiding more details than may be needed by anything but the friends of the class.

Finally, friends are particularly common in cases of operator overloading because it is often necessary for an overloaded operator to have access to the internals of the classes that are arguments to the operator.

