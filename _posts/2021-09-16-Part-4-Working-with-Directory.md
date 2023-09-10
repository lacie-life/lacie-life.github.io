---
title: Working with Directory
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2021-09-16 11:11:11 +0700
categories: [Linux]
tags: [Tutorial]
render_with_liquid: false
---

### Working with Directory

1. Linux Special Directories:

. - The Current Directory

.. - The Parent Directory

/ - The Root Directory

~ - The Current User's Home Directory

2. Absolute Path vs Relative Path:

absolute path is defined as specifying the location of a file or directory from the root directory(/).

Relative path is defined as the path related to the present working directly.

3. Directories Commands:

cd - change directory to

cd .. - change directory to the parent directory

cd - - change directory to the previous Directory (works like back)

pwd - Print Working Directory (Full Absolute Path)

mkdir - Make a new directory

mkdir -p - Make a new directory (with -p parent) as if parent doesn't exist create as well

rmdir - remove a directory

rmdir -p - remove a directory (with -p parent) as delete with parent as well

rm -rf - remove a directory (using the rm command with recursive and force options)

4. Listing Commands

ls - list Files

ls -l - list Files with Long List Format

ls -t - list Files with sorting Last Modified time

ls -h - list Files with Human readable format (1000 = 1K)

ls -a - list All Files (hidden included) Hidden Files start with Dot (.) --> e.g: .ssh

ls -laht - list All Files (hidden included) and sort Last Modified time and with Human readable with Long List Format

