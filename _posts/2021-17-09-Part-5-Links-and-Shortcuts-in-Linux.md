### Links and Shortcuts in Linux

1. commands:

ln - create a hard link

ln -s - create a symbolic (Soft) link (the same as Shortcuts in Windows)

2. Hard links Vs Symbolic links

Hard links and symbolic links are two different methods to refer to a file in the hard drive.

hard link is essentially a synced carbon copy of a file that refers directly to the inode of a file.

Symbolic links on the other hand refer directly to the file which refers to the inode, (the same as Shortcut in Windows).

3. What is inode ?

inode (index node) is a data structure in a Unix-style file system that describes a file-system object such as a file or a directory. Each inode stores the attributes and disk block locations of the object's data.File-system object attributes may include metadata (times of last change, access, modification), as well as owner and permission data.

