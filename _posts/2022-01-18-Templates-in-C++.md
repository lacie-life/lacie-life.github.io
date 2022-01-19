# Templates in C++

A template is a simple and yet very powerful tool in C++. The simple idea is to pass data type as a parameter so that we don’t need to write the same code for different data types. For example, a software company may need sort() for different data types. Rather than writing and maintaining the multiple codes, we can write one sort() and pass data type as a parameter. 

C++ adds two new keywords to support templates: ‘template’ and ‘typename’. The second keyword can always be replaced by keyword ‘class’.

## How do templates work? 

Templates are expanded at compiler time. This is like macros. The difference is, the compiler does type checking before template expansion. The idea is simple, source code contains only function/class, but compiled code may contain multiple copies of same function/class. 

![Fig.1](https://media.geeksforgeeks.org/wp-content/cdn-uploads/gq/2014/06/templates-cpp.jpg)

Function Templates We write a generic function that can be used for different data types. Examples of function templates are sort(), max(), min(), printArray(). 

```
#include <iostream>
using namespace std;

// One function works for all data types. This would work
// even for user defined types if operator '>' is overloaded
template <typename T>
T myMax(T x, T y)
{
return (x > y)? x: y;
}

int main()
{
cout << myMax<int>(3, 7) << endl; // Call myMax for int
cout << myMax<double>(3.0, 7.0) << endl; // call myMax for double
cout << myMax<char>('g', 'e') << endl; // call myMax for char

return 0;
}
```

Output:

```
7
7
g
```

Below is the program to implement Bubble Sort using templates in C++: 

```
// CPP code for bubble sort
// using template function
#include <iostream>
using namespace std;

// A template function to implement bubble sort.
// We can use this for any data type that supports
// comparison operator < and swap works for it.
template <class T>
void bubbleSort(T a[], int n) {
	for (int i = 0; i < n - 1; i++)
		for (int j = n - 1; i < j; j--)
			if (a[j] < a[j - 1])
			swap(a[j], a[j - 1]);
}

// Driver Code
int main() {
	int a[5] = {10, 50, 30, 40, 20};
	int n = sizeof(a) / sizeof(a[0]);

	// calls template function
	bubbleSort<int>(a, n);

	cout << " Sorted array : ";
	for (int i = 0; i < n; i++)
		cout << a[i] << " ";
	cout << endl;

return 0;
}
```

Output:

```
Sorted array : 10 20 30 40 50
```

Class Templates Like function templates, class templates are useful when a class defines something that is independent of the data type. Can be useful for classes like LinkedList, BinaryTree, Stack, Queue, Array, etc. 
Following is a simple example of template Array class.

```
#include <iostream>
using namespace std;

template <typename T>
class Array {
private:
	T *ptr;
	int size;
public:
	Array(T arr[], int s);
	void print();
};

template <typename T>
Array<T>::Array(T arr[], int s) {
	ptr = new T[s];
	size = s;
	for(int i = 0; i < size; i++)
		ptr[i] = arr[i];
}

template <typename T>
void Array<T>::print() {
	for (int i = 0; i < size; i++)
		cout<<" "<<*(ptr + i);
	cout<<endl;
}

int main() {
	int arr[5] = {1, 2, 3, 4, 5};
	Array<int> a(arr, 5);
	a.print();
	return 0;
}
```

Output:

```
1 2 3 4 5
```

## Can there be more than one arguments to templates? 

Yes, like normal parameters, we can pass more than one data types as arguments to templates. The following example demonstrates the same. 

```
#include<iostream>
using namespace std;

template<class T, class U>
class A {
	T x;
	U y;
public:
	A() { cout<<"Constructor Called"<<endl; }
};

int main() {
    A<char, char> a;
    A<int, double> b;
    return 0;
}
```

Output:

```
Constructor Called
Constructor Called
```

## Can we specify default value for template arguments? 

Yes, like normal parameters, we can specify default arguments to templates. The following example demonstrates the same. 

```
#include<iostream>
using namespace std;

template<class T, class U = char>
class A {
public:
	T x;
	U y;
	A() { cout<<"Constructor Called"<<endl; }
};

int main() {
    A<char> a; // This will call A<char, char>
    return 0;
}
```

Output:

```
Constructor Called
```

## What is the difference between function overloading and templates? 
Both function overloading and templates are examples of polymorphism feature of OOP. Function overloading is used when multiple functions do similar operations, templates are used when multiple functions do identical operations.

## What happens when there is a static member in a template class/function? 
Each instance of a template contains its own static variable. See about Templates and Static variables for more details.

## What is template specialization? 
Template specialization allows us to have different code for a particular data type. See about Template Specialization for more details.

## Can we pass nontype parameters to templates? 
We can pass non-type arguments to templates. Non-type parameters are mainly used for specifying max or min values or any other constant value for a particular instance of a template. The important thing to note about non-type parameters is, they must be const. The compiler must know the value of non-type parameters at compile time. Because the compiler needs to create functions/classes for a specified non-type value at compile time. In below program, if we replace 10000 or 25 with a variable, we get a compiler error. Please see this.
Below is a C++ program. 

```
// A C++ program to demonstrate working of non-type
// parameters to templates in C++.
#include <iostream>
using namespace std;

template <class T, int max>
int arrMin(T arr[], int n)
{
    int m = max;
    for (int i = 0; i < n; i++)
	    if (arr[i] < m)
		    m = arr[i];

    return m;
}

int main()
{
    int arr1[] = {10, 20, 15, 12};
    int n1 = sizeof(arr1)/sizeof(arr1[0]);

    char arr2[] = {1, 2, 3};
    int n2 = sizeof(arr2)/sizeof(arr2[0]);

    // Second template parameter to arrMin must be a constant
    cout << arrMin<int, 10000>(arr1, n1) << endl;
    cout << arrMin<char, 256>(arr2, n2);
    return 0;
}

```

output:
```
10
1
```

## Template Metaprogramming

Predict the output of following C++ program.

```
#include <iostream>
using namespace std;

template<int n> struct funStruct
{
	enum { val = 2*funStruct<n-1>::val };
};

template<> struct funStruct<0>
{
	enum { val = 1 };
};

int main()
{
	cout << funStruct<8>::val << endl;
	return 0;
}

```

Output:

```
256
```

The program calculates “2 raise to the power 8 (or 2^8)”. In fact, the structure funStruct can be used to calculate 2^n for any known n (or constant n). The special thing about above program is: calculation is done at compile time. So, it is compiler that calculates 2^8. To understand how compiler does this, let us consider the following facts about templates and enums:
1) We can pass nontype parameters (parameters that are not data types) to class/function templates. 
2) Like other const expressions, values of enumeration constants are evaluated at compile time. 
3) When compiler sees a new argument to a template, compiler creates a new instance of the template.
Let us take a closer look at the original program. When compiler sees funStruct<8>::val, it tries to create an instance of funStruct with parameter as 8, it turns out that funStruct<7> must also be created as enumeration constant val must be evaluated at compile time. For funStruct<7>, compiler need funStruct<6> and so on. Finally, compiler uses funStruct<1>::val and compile time recursion terminates. So, using templates, we can write programs that do computation at compile time, such programs are called template metaprograms. Template metaprogramming is in fact Turing-complete, meaning that any computation expressible by a computer program can be computed, in some form, by a template metaprogram. Template Metaprogramming is generally not used in practical programs, it is an interesting concept though. 

