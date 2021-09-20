### Shell Script

Shell script is a list of commands in a computer program that is run by the Unix shell which is a command line interpreter. A shell script usually has comments that describe the steps.

Important Things About Shell Scripting:

1. Variables and Special Variables

$0 - The filename of the current script.
$n - The Nth argument passed to script was invoked or function was called.
$# - The number of argument passed to script or function.
$@ - All arguments passed to script or function.
$* - All arguments passed to script or function.
$? - The exit status of the last command executed.
$$ - The process ID of the current shell. For shell scripts, this is the process ID under which they are executing.
$! - The process number of the last background command.

2. Operators

```
# Shell operator
VAR=`expr 1 + 1`
echo $VAR
```

3. If ... else

```
a="10"
b="20"

if [ $a == $b ]
then 
    echo "a is equal to b"
fi

if [ $a != $b ]
then 
    echo "a is not equal to b"
fi
```

4. Case ... esac

```
FRUIT="apple"

case "$FRUIT" in
    "apple") echo "Apple"
    ;;
    "baana") echo "Banana"
    ;;
    "mango") echo "????"
    ;;
esac
```

4. Loops

#### While loops
```
a=0
while [ "$a" -lt 10 ]  
do
    b="$a"
    while [ "$b" -ge 0 ]
    do 
        echo -n "$b "
        b = `expr $b - 1`
    done
    echo 
    a=`expr $a + 1`
done
```

#### For loops

```
for var1 in 1 2 3
do
    for var2 in 0 5
    do  
        if [ $var1 -eq 2 -a $var2 -eq 0 ]
        then 
            break 2
        else
            echo "$var1 $var2"
        fi
    done
done
```

5. Functions

```
# Define function 
Hello () {
    echo "Hello World"
}

# Invoke function
Hello
```
