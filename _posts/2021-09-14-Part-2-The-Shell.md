---
title: The Shell Theory
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2021-09-14 11:11:11 +0700
categories: [Linux]
tags: [Tutorial]
render_with_liquid: false
---

### The Shell Theory

1. What is The Shell ?

A Shell provides you with an interface to the Unix system. It gathers input from you and executes programs based on that input. When a program finishes executing, it displays that program's output.

Shell is an environment in which we can run our commands, programs, and shell scripts. There are different flavors of a shell, just as there are different flavors of operating systems. Each flavor of shell has its own set of recognized commands and functions.

2. What is Shell Prompt ?

The prompt, $, which is called the command prompt, is issued by the shell. While the prompt is displayed, you can type a command.

Shell reads your input after you press Enter. It determines the command you want executed by looking at the first word of your input. A word is an unbroken set of characters. Spaces and tabs separate words.

Examples:
```
ubuntu@My-Computer:~$

root@ip-192-168-1-206:~$
```

3. What is This Tilde (~) ?

The tilde (~) is a Linux "shortcut" to denote a user's home directory. Thus tilde slash (~/) is the beginning of a path to a file or directory below the user's home directory.

For example, for the ubuntu user, file /home/ubuntu/test.file can also be denoted by ~/test.file

Note: this if you are currently logged in as Ubuntu user so ~ will refer to the home directory of ubuntu

but if you are logged in as another user for example root it will refer to the root's Home directory which is /root

What is the SuperUser (root) ?

Superuser is the generic term to refer to the user account used for system administration (Full Permission).

In Linux operating systems the superuser is the account whose UID (user identifier) is zero, no matter how it is named. and it's named root by default.

