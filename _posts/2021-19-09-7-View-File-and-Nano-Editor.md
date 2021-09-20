### Viewing Files and Editor

1. Basic commands :

cat t.txt - display the content of the t.txt file

more t.txt - Browse through the t.txt file

less t.txt - with Features more than more command

head t.txt - Output the beginning of the t.txt file

head -15 t.txt - Output the beginning of the t.txt file First 15 Lines

tail t.txt - Output the ending of the t.txt file

tail -15 t.txt - Output the ending of the t.txt file Last 15 Lines

tail -f t.txt - Output the ending of the t.txt file as it is being written to

nano - open and use Nano Editor

2. Vi vs Vim vs View

Vi stands for Visual. It is a text editor that is an early attempt to a visual text editor.

Vim stands for Vi IMproved. It is an implementation of the Vi standard with many additions

View stars Vim with Real-only mode

#### Most Important Vi / Vim Shortcuts We Will Use :

^ (shift + 6) - Go to the Beginning of the line

$ (shift + 4) - Go to the End of the line

i - insert at the cursor position

I (shift + i) - insert at the beginning of the line

a - append after the cursor position

A (shift + a) - append at the end of the line

o - insert a new empty line below the cursor position

:w - writes (saves) the file

:w! - Forces the file to be saved

:q - Quit

:q! - quit without saving any thing

:wq - Write and Quit

:wq! - Write and Quit Forcefully

:x - same as :wq

:n - Positions the cursor at line n (e.g: :1 , :4 , :146)

:$ - Positions the cursor at the last line

gg - Positions the cursor at the First line of the file

G (shift + g) - Positions the cursor at the last line of the file

:set nu - Turn on line numbering

:set nonu - Turn off line numbering

:help [subcommand] - get help for that subcommand

v - Visual Selection

y (stands for yank) - copy

d (stands for delete) - Cut

p - Paste

yy - copy the current line (use without v)

u - Undo

/<Pattern> - to search for the specified Pattern

n - after typing enter for the /<Pattern> then n will search for the next

N - after typing enter for the /<Pattern> then N will search for the Previous
