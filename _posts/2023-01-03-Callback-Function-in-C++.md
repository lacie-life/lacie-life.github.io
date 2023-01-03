---
title: Designing Callbacks in C++
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-01-03 11:11:14 +0700
categories: [Theory]
tags: [Skill]
img_path: /assets/img/post_assest/
render_with_liquid: false
---

In this article we will discuss what is a callback and what types of callbacks we can have in C++ and how to Design Callbacks it.

## Now what is a callback?

Callback is a function that we pass to another APIs as argument while calling them. Now these APIs are expected to use our provided callback at some point.

## Callbacks in C++ can be of 3 types

1. Function Pointer

2. Function Objects / Functors

3. Lambda functions

Behaviour or result of the API is dependent on the callback we provide i.e. if we keep the input to API same and just change the callback then the output of API will change. 

### Using Function Pointer as callback

From a framework we got an API that can build complete message from provided raw data. This API will perform following steps,

- Add header and footer in raw Data to make the message.
- Encrypt the complete message.
- Return the message

Now this API knows what header and footer to apply but don’t know about exact encryption logic, because that is not fixed to Framework, that can change from application to application. As, Encryption Logic is application dependent, therefore in this API provides a provision to pass encryption logic as a callback function pointer. This API will call back the code of application by calling this passed function pointer.

![Fig. 1: Callback Mechanism](https://thispointer.com/wp-content/uploads/2015/07/callback.png)

Framework’s API that accepts the function pointer callback as an argument is as follows,

```
std::string buildCompleteMessage(std::string rawData, std::string (* encrypterFunPtr)(std::string) )
{
    // Add some header and footer to data to make it complete message
    rawData = "[HEADER]" + rawData + "[FooTER]";
    // Call the callBack provided i.e. function pointer to encrypt the
    rawData = encrypterFunPtr(rawData);
    return rawData;
}

```

Now when an application calls this API of framework to build the message, it passes a pointer to its local function as an argument.

In this case this a function Pointer is a Callback,

```
//This encrypt function increment all letters in string by 1.
std::string encryptDataByLetterInc(std::string data)
{
    for(int i = 0; i < data.size(); i++)
    {
        if( (data[i] >= 'a' && data[i] <= 'z' ) || (data[i] >= 'A' && data[i] <= 'Z' ) )
            data[i]++;
    }
    return data;
}
```

This encrypt function increment all letters in string by 1.

Now lets call build message API will this function as callback,

```
std::string msg = buildCompleteMessage("SampleString", &encryptDataByLetterInc);
    std::cout<<msg<<std::endl;
```

<i>
Output:

[IFBEFS]TbnqmfTusjoh[GppUFS]

</i>

Now, lets design a new type of encryption function

```
//This encrypt function decrement all letters in string by 1.
std::string encryptDataByLetterDec(std::string data)
{
    for(int i = 0; i < data.size(); i++)
    {
        if( (data[i] >= 'a' && data[i] <= 'z' ) || (data[i] >= 'A' && data[i] <= 'Z' ) )
            data[i]--;
    }
    return data;
}
```

This encrypt function decrement all letters in string by 1. Now lets call build message API will this function as callback,

```
std::string msg = buildCompleteMessage("SampleString", &encryptDataByLetterDec);
std::cout<<msg<<std::endl;
```

<i>
Output:

[GD@CDQ]R`lokdRsqhmf[EnnSDQ]
</i>

Now output of this call  is different to earlier.
We kept the API and passed data same but changed the passed callback, as a result behaviour or result of API changed.

This shows, that behaviour or result of the API is dependent on the callback passed as an argument.

### Function Objects & Functors

#### What is a Function Objects

A Function Object / Functor is a kind of Callback with State.

Object of a class which has overloaded operator() is called Function Object or Functor i.e. a class with overloaded operator() function is as follows,

```
#include <iostream>
class MyFunctor
{
public:
    int operator()(int a , int b)
    {
        return a+b;
    }
};
```

Now lets create a function object and call them,

```
MyFunctor funObj;
std::cout<<funObj(2,3)<<std::endl;
```

Function objects / Functors can be called just like normal functions i.e.

```
MyFunctor funObj;
funObj(2,3);
```

is similar to following code,

```
MyFunctor funObj;
funObj.operator ()(2,3);
```

Let’s understand by a practice example,

In previous part we discussed how to use function pointer call back to change the behaviour of message builder API. 

1. Let’s refresh the problem statement,

From a framework we got an API that can build complete message from provided raw data. This API will perform following steps,

- Add header and footer in raw Data to make the message.
- Encrypt the complete message.
- Return the message

Now this API knows what header and footer to apply but don’t know about exact encryption logic, because that is not fixed to Framework, that can change from application to application.
As, Encryption Logic is application dependent, therefore it provides a provision to pass encryption logic as a callback function pointer,

```
std::string buildCompleteMessage(std::string rawData,
        std::string (*encrypterFunPtr)(std::string)) {
    // Add some header and footer to data to make it complete message
    rawData = "[HEADER]" + rawData + "[FooTER]";
    // Call the callBack provided i.e. function pointer to encrypt the
    rawData = encrypterFunPtr(rawData);
    return rawData;
}
```

No suppose in our application we want to call this framework API three times with 3 different types of encryption logics i.e.

<i>
Encrypt by incrementing each letter by 1.
Encrypt by incrementing each letter by 2.
Encrypt by decrementing each letter by 1.
</i>

2. How to this through function pointers

For this we need to create three different types of functions and use them as function pointers while calling this API.
3 different functions here are necessary because normal global functions don’t have any state associated with them,

```
//This encrypt function increment all letters in string by 1.
std::string encryptDataByLetterInc1(std::string data) {
    for (int i = 0; i < data.size(); i++)
        if ((data[i] >= 'a' && data[i] <= 'z')
                || (data[i] >= 'A' && data[i] <= 'Z'))
            data[i]++;
    return data;
}
//This encrypt function increment all letters in string by 2.
std::string encryptDataByLetterInc2(std::string data) {
    for (int i = 0; i < data.size(); i++)
        if ((data[i] >= 'a' && data[i] <= 'z')
                || (data[i] >= 'A' && data[i] <= 'Z'))
            data[i] = data[i] + 2;
    return data;
}
//This encrypt function increment all letters in string by 1.
std::string encryptDataByLetterDec(std::string data) {
    for (int i = 0; i < data.size(); i++)
        if ((data[i] >= 'a' && data[i] <= 'z')
                || (data[i] >= 'A' && data[i] <= 'Z'))
            data[i] = data[i] - 1;
    return data;
}
int main() {
    std::string msg = buildCompleteMessage("SampleString",
            &encryptDataByLetterInc1);
    std::cout << msg << std::endl;
    msg = buildCompleteMessage("SampleString", &encryptDataByLetterInc2);
    std::cout << msg << std::endl;
    msg = buildCompleteMessage("SampleString", &encryptDataByLetterDec);
    std::cout << msg << std::endl;
    return 0;
}
```

Is there any way to bind state with function pointers?

Answers is yes –  By making the them function objects or functors.

3. Let’s create a Function Object / Functor for Encryption:

A function object or functor, is an object that has operator () defined as member function i.e.

```
class Encryptor {
    bool m_isIncremental;
    int m_count;
public:
    Encryptor() {
        m_isIncremental = 0;
        m_count = 1;
    }
    Encryptor(bool isInc, int count) {
        m_isIncremental = isInc;
        m_count = count;
    }
    std::string operator()(std::string data) {
        for (int i = 0; i < data.size(); i++)
            if ((data[i] >= 'a' && data[i] <= 'z')
                    || (data[i] >= 'A' && data[i] <= 'Z'))
                if (m_isIncremental)
                    data[i] = data[i] + m_count;
                else
                    data[i] = data[i] - m_count;
        return data;
    }
};
```

These function objects can be called just like functions in message builder API.

```
std::string buildCompleteMessage(std::string rawData,
        Encryptor encyptorFuncObj) {
    // Add some header and footer to data to make it complete message
    rawData = "[HEADER]" + rawData + "[FooTER]";
    // Call the callBack provided i.e. function pointer to encrypt the
    rawData = encyptorFuncObj(rawData);
    return rawData;
}
```

What just happened here is a temporary object of EncruptFunctor is created and operator () is called on it.

It can also be called like,

```
Encryptor(true, 1);
```

It similar to calling,

```
Encryptor tempObj;
tempObj.operator(true, 1);
```

Now use this function object two solve our problem. Create a function object that can encrypt by decementing / increment each letter by specified number of times.

```
int main() {
    std::string msg = buildCompleteMessage("SampleString", Encryptor(true, 1));
    std::cout << msg << std::endl;
    msg = buildCompleteMessage("SampleString", Encryptor(true, 2));
    std::cout << msg << std::endl;
    msg = buildCompleteMessage("SampleString", Encryptor(false, 1));
    std::cout << msg << std::endl;
    return 0;
}
```

So, this is how we can use function objects as callbacks.

Complete executable code is as follows,

```
#include <iostream>

class Encryptor {
    bool m_isIncremental;
    int m_count;
public:
    Encryptor() {
        m_isIncremental = 0;
        m_count = 1;
    }
    Encryptor(bool isInc, int count) {
        m_isIncremental = isInc;
        m_count = count;
    }
    std::string operator()(std::string data) {
        for (int i = 0; i < data.size(); i++)
            if ((data[i] >= ‘a’ && data[i] <= ‘z’)
                    || (data[i] >= ‘A’ && data[i] <= ‘Z’))
                if (m_isIncremental)
                    data[i] = data[i] + m_count;
                else
                    data[i] = data[i] – m_count;
        return data;
    }

};

std::string buildCompleteMessage(std::string rawData,
        Encryptor encyptorFuncObj) {
    // Add some header and footer to data to make it complete message
    rawData = "[HEADER]" + rawData + "[FooTER]";

    // Call the callBack provided i.e. function pointer to encrypt the
    rawData = encyptorFuncObj(rawData);

    return rawData;
}

int main() {
    std::string msg = buildCompleteMessage("SampleString", Encryptor(true, 1));
    std::cout << msg << std::endl;

    msg = buildCompleteMessage("SampleString", Encryptor(true, 2));
    std::cout << msg << std::endl;

    msg = buildCompleteMessage("SampleString", Encryptor(false, 1));
    std::cout << msg << std::endl;

    return 0;
}

```

### C++11 Lambda Function

#### What is a Lambda Function?

Lambda functions are a kind of anonymous functions in C++. These are mainly used as callbacks in C++. Lambda function is like a normal function i.e.

- You can pass arguments to it
- It can return the result

But it doesn’t have any name. Its mainly used when we have to create very small functions to pass as a callback to an another API.

Before going deep into lambda functions, lets understand what was the need of lambda functions.

#### Need of Lambda functions

I have an array of int and I want to traverse on this array and print all ints using STL algorithm std::for_each.

Let’s do that using a function pointer,

```
#include <iostream>
#include <algorithm>
void display(int a)
{
    std::cout<<a<<" ";
}
int main() {
    int arr[] = { 1, 2, 3, 4, 5 };
    std::for_each(arr, arr + sizeof(arr) / sizeof(int), &display);
    std::cout << std::endl;
}
```

In this above example, to do a simple display we created a separate function. Is there any way through which we can achieve our requirement and also avoid this overhead.

#### Rise of Lambda functions

To resolve this, use we can use lambda functions. A lambda function is a kind of anonymous function which doesn’t have any name but you can pass arguments and return results from it. Also all its content will work as in-line code.

Lambda function example is as follows,

```
[](int x) {
        std::cout<<x<<" ";
    }
```

Here,
- [] is used to pass the outer scope elements
- (int x) shows argument x is passed

Let’s see the above example with lambda functions,

```
#include <iostream>
#include <algorithm>
int main() {
    int arr[] = { 1, 2, 3, 4, 5 };
    std::for_each(arr, arr + sizeof(arr) / sizeof(int), [](int x) {
            std::cout<<x<<" ";
        });
    std::cout << std::endl;
}
```

#### How to pass outer scope elements inside lambda functions

Case 1: Using [=]

```
    [=](int &x) {
        // All outer scope elements has been passed by value
    }
```

Case 2: Using [&]

```
    [&](int &x) {
        // All outer scope elements has been passed by reference
    }
```

Checkout this example this clearly shows how to use outer scope elements inside the lambda functions.

```
#include <iostream>
#include <algorithm>
int main() {
    int arr[] = { 1, 2, 3, 4, 5 };
    int mul = 5;
    std::for_each(arr, arr + sizeof(arr) / sizeof(int), [&](int x) {
        std::cout<<x<<" ";
        // Can modify the mul inside this lambda function because
        // all outer scope elements has write access here.
            mul = 3;
        });
    std::cout << std::endl;
    std::for_each(arr, arr + sizeof(arr) / sizeof(int), [=](int &x) {
        x= x*mul;
        // Can not modify the mul inside this lambda function because
        // all outer scope elements has read only access here.
        // mul = 9;
        });
    std::cout << std::endl;
    std::for_each(arr, arr + sizeof(arr) / sizeof(int), [](int x) {
        // No access to mul inside this lambda function because
        // all outer scope elements are not visible here.
        //std::cout<<mul<<" ";
        });
    std::cout << std::endl;
}
```
