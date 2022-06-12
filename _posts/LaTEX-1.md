---
title: Learn LaTEX - [Part 1]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-05-21 11:11:11 +0700
categories: [Skill]
tags: [tutorial]
img_path: /assets/img/post_assest/
render_with_liquid: false
---

In this guide, we hope to give you your first introduction to LaTEX. The guide does not require you to have any prior knowledge of LaTEX, but by the time you are finished, you will have written your first LaTeX document, and hopefully will have a good knowledge of some of the basic functions provided by LaTEX.

## What is LaTEX

LATEX (pronounced LAY-tek or LAH-tek) is a tool used to create professional-looking documents. It is based on the WYSIWYM (what you see is what you mean) idea, meaning you only have focus on the contents of your document and the computer will take care of the formatting. Instead of spacing out text on a page to control formatting, as with Microsoft Word or LibreOffice Writer, users can enter plain text and let LATEX take care of the rest.

## Why learn LaTEX?

LaTEX is used all over the world for scientific documents, books, as well as many other forms of publishing. Not only can it create beautifully typeset documents, but it allows users to very quickly tackle the more complicated parts of typesetting, such as inputting mathematics, creating tables of contents, referencing and creating bibliographies, and having a consistent layout across all sections. Due to the huge number of open source packages available (more on this later), the possibilities with LaTEX are endless. These packages allow users to do even more with LaTEX, such as add footnotes, draw schematics, create tables etc.

One of the most important reasons people use LaTEX is that it separates the content of the document from the style. This means that once you have written the content of your document, we can change its appearance with ease. Similarly, you can create one style of document which can be used to standardise the appearance of many different documents. This allows scientific journals to create templates for submissions. These templates have a pre-made layout meaning that only the content needs to be added. In fact there are hundreds of templates available for everything from CVs to slideshows.

## Writing your first piece of LATEX

The first step is to create a new LATEX project. You can do this on your own computer by creating a new .tex file, or else you can start a new project in Overleaf. Let's start with the simplest working example:

```
\documentclass{article}

\begin{document}
First document. This is a simple example, with no 
extra parameters or packages included.
\end{document}
```

![Fig.1](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/0/01/Firstdocsmall.PNG)

You can see that LaTEX has already taken care of the first piece of formatting for you, by indenting the first line of the paragraph. Let's have a close look at what each part of our code does.

The first line of code declares the type of document, known as the <i> class </i>. The class controls the overall appearance of the document. Different types of documents will require different classes i.e. a CV/resume will require a different class than a scientific paper. In this case, the class is <b> article </b>, the simplest and most common LATEX class. Other types of documents you may be working on may require different classes such as <b> book </b> or <b> report </b>.

After this, you write the content of our document, enclosed inside the <b> \begin{document} </b> and <b> \end{document} </b> tags. This is known as the body of the document. You can start writing here and make changes to the text if you wish. To see the result of these changes in the PDF you have to compile the document. To do this in Overleaf, simply hit <i> <b> Recompile </b> </i>. (You can also set your project to automatically recompile when you edit your files, by clicking on the small arrow next to the 'Recompile button and set 'Auto Compile to 'On.)

If you are using a basic text editor such as gedit, emacs, vim, sublime, notepad etc., you will have to compile the document manually. To do this, simply run pdflatex <your document> in your computers terminal/command line. See here for more information on how to do this.

If you are using a dedicated LaTeX editor such as TeXmaker or TeXworks, simply hit the Recompile button. Consult the programs documentation if you are unsure of where this is.

Now that you have learnt how to add content to our document, the next step is to give it a title. To do this, we must talk briefly about the preamble.

## The preamble of a document

In the previous example the text was entered after the <b> \begin{document} </b> command. Everything in your <b> .tex </b> file before this point is called the preamble. In the preamble you define the type of document you are writing, the language you are writing in, the packages you would like to use (more on this later) and several other elements. For instance, a normal document preamble would look like this:

```
\documentclass[12pt, letterpaper]{article}
\usepackage[utf8]{inputenc}
```

Below a detailed description of each line:

<b> \documentclass[12pt, letterpaper]{article} </b>
As said before, this defines the type of document. Some additional parameters included in the square brackets can be passed to the command. These parameters must be comma-separated. In the example, the extra parameters set the font size (12pt) and the paper size (letterpaper). Of course other font sizes (9pt, 11pt, 12pt) can be used, but if none is specified, the default size is 10pt. As for the paper size other possible values are a4paper and legalpaper; see the article about Page size and margins for more details about this.

<b> \usepackage[utf8]{inputenc} </b>

This is the encoding for the document. It can be omitted or changed to another encoding but utf-8 is recommended. Unless you specifically need another encoding, or if you are unsure about it, add this line to the preamble.

## Adding a title, author and date

To add a title, author and date to our document, you must add three lines to the preamble (NOT the main body of the document). These lines are

<b> \title{First document} </b>

This is the title.

<b> \author{Hubert Farnsworth} </b>

Here you put the name of the Author(s) and, as an optional addition, you can add the next command within the curly braces:

<b> \thanks{funded by the Overleaf team} </b>

This can be added after the name of the author, inside the braces of the author command. It will add a superscript and a footnote with the text inside the braces. Useful if you need to thank an institution in your article.

<b> \date{February 2014} </b>

You can enter the date manually or use the command \today so the date will be updated automatically at the time you compile your document

With these lines added, your preamble should look something like this

```
\documentclass[12pt, letterpaper, twoside]{article}
\usepackage[utf8]{inputenc}

\title{First document}
\author{Hubert Farnsworth \thanks{funded by the Overleaf team}}
\date{February 2017}
```

Now that you have given your document a title, author and date, you can print this information on the document with the \maketitle command. This should be included in the body of the document at the place you want the title to be printed.

```
\begin{document}

\maketitle

We have now added a title, author and date to our first \LaTeX{} document!

\end{document}
```

![Fig.2](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/e/e9/Learnlatex1.PNG)

## Adding comments

As with any code you are writing, it can often be useful to include comments. Comments are pieces of text you can include in the document which will not be printed, and will not affect the document in any way. They are useful for organizing your work, taking notes, or commenting out lines/sections when debugging. To make a comment in LATEX, simply write a % symbol at the beginning of the line as shown below:

```
\begin{document}

\maketitle

We have now added a title, author and date to our first \LaTeX{} document!

% This line here is a comment. It will not be printed in the document.

\end{document}
```

![Fig.3](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/e/e9/Learnlatex1.PNG)

## Bold, italics and underlining

We will now look at some simple text formatting commands.

- Bold: Bold text in LaTeX is written with the \textbf{...} command.
- Italics: Italicised text in LaTeX is written with the \textit{...} command.
- Underline: Underlined text in LaTeX is written with the \underline{...} command.

An example of each of these in action is shown below:

```
Some of the \textbf{greatest}
discoveries in \underline{science} 
were made by \textbf{\textit{accident}}.
```

![Fig.4](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/a/a9/Biu1.png)

Another very useful command is the \emph{...} command. What the \emph command actually does with its argument depends on the context - inside normal text the emphasized text is italicized, but this behaviour is reversed if used inside an italicized text- see example below:

```
Some of the greatest \emph{discoveries} 
in science 
were made by accident.

\textit{Some of the greatest \emph{discoveries} 
in science 
were made by accident.}

\textbf{Some of the greatest \emph{discoveries} 
in science 
were made by accident.}
```

![Fig.5](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/5/5d/Biu5.png)

Moreover, some packages, e.g. Beamer, change the behaviour of \emph command.

## Adding images

We will now look at how to add images to a LATEX document. On Overleaf, you will first have to upload the images.

Below is a example on how to include a picture.

```
\documentclass{article}
\usepackage{graphicx}
\graphicspath{ {images/} }

\begin{document}
The universe is immense and it seems to be homogeneous, 
in a large scale, everywhere we look at.

\includegraphics{universe}

There's a picture of a galaxy above
\end{document}
```

![Fig.6](https://sharelatex-wiki-cdn-671420.c.cdn77.org/learn-scripts/images/9/9d/InsertingImagesEx1.png)

LATEX can not manage images by itself, so you will need to use a package. Packages can be used to change the default look of your LATEX document, or to allow more functionalities. In this case, you need to include an image in our document, so you should use the <b> graphicx </b> package. This package gives new commands, <b> \includegraphics{...} </b> and <b> \graphicspath{...} </b>. To use the graphicx package, include the following line in you preamble: <b> \usepackage{graphicx} </b>

The <b> \includegraphics{universe} </b> command is the one that actually included the image in the document. Here universe is the name of the file containing the image without the extension, then universe.PNG becomes universe. The file name of the image should not contain white spaces nor multiple dots.



