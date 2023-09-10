---
title: Input Output and Redirection
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2021-09-18 11:11:11 +0700
categories: [Linux]
tags: [Tutorial]
render_with_liquid: false
---

### Input Output and Redirection

1. What is File Descriptor?

In Unix and related computer operating systems, a file descriptor (FD) is an abstract indicator used to access a file or other input/output resource, such as a pipe or network socket. File descriptors form part of the POSIX application programming interface.

2. Input/Output Types

I/O Name Abbreviation File Descriptor

Standard Input stdin 0

Standard Output stdout 1

Standard Error stderr 2

3. Redirection

``` > ``` Redirects standard output to a file (Overwrite existing content)

``` >> ```Redirects standard output to a file (Append to any existing content)

``` < ``` Redirects input from a file to a command

4. Example

ls -l > file.txt - this will put Output of ls command to file.txt (with overwriting)

2 > /dev/null - this will put any error happens to the null place

sort < file.txt - this will take the input from the file.txt for the sort command

5. Transfer File Over Network

use scp command


