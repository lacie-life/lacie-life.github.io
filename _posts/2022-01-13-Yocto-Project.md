---
title: Yocto Project
author:
  name: Life Zero
  link: https://github.com/lacie-life
date:  2022-01-12 11:11:11 +0700
categories: [Linux, Yocto]
tags: [wrtting]
render_with_liquid: false
---

# Yocto Project

Yocto Project is a cool open-source community project that allows you to build your own Linux system from scratch to booting into your embedded hardware, which makes embedded Linux simple and accessible. It helps developers create customized systems based on the Linux kernel by providing useful templates, tools, and methods, and wide hardware architectures support on ARM, PPC, MIPS, x86 (32 & 64 bit).

This article will talk about:

- Yocto Project basics including structure and essential terminology.

- How Yocto Project helps embedded hardware systems customized from scratch to a final consumer product

![Fig.1](https://www.seeedstudio.com/blog/wp-content/uploads/2021/09/image-22.png)

## What is Yocto Project?

The Yocto Project is structured with cross-platform tools, metadata, to enable developers rapidly to create customized Linux distributions from source code, which simplifies the development process. Compared to a full Linux distribution, the customized system will reserve the software you need to make the system much more specific to your application. Yocto Project has advantages in system and application development, archiving, and management. Developers can customize their systems in terms of speed, memory footprint, and even memory utilization. Yocto Project allows software customization and construction exchange for multiple hardware platforms, and also maintains software stack in scale.

Yocto Project provides extensive flexibility. For enterprise businesses, embedded developers can customize internal Linux Distros based on basic existing outstanding Linux systems and deploy them into multiple final products.

With the help of Yocto on your operating system development, it will reduce the customization burden of crossed Linux Distros upgrades and help reduce the required work for cross-architecture migration.

## What are Yocto Project components, tools and workflow of a build?

Yocto Project is a combination of open-source projects and metadata and aims to help developers develop customized Linux systems for embedded products, regardless of the hardware architecture. Yocto Project provides a series of tools and abundant resources powered (including continuation, software, configuration, templates) community to create (tailored) Linux systems for embedded devices.

The components and tools are separate from the reference distribution (Poky) and the OpenEmbedded build system. Most components and tools are downloaded separately. Yocto Project combines, maintains, and verifies the following key components:

![Fig.2](https://www.seeedstudio.com/blog/wp-content/uploads/2021/09/image-18.png)

1. BitBake: The core tool of the OpenEmbedded build system.

BitBake plays the role of build system engine and is responsible for parsing metadata, generating task lists from it, and then performing these tasks.

2. OpenEmbedded build system which is jointly maintained with OpenEmbedded Project.
OpenEmbedded-Core (OE-Core) is metadata composed of basic recipes, classes, and related files. These metadata are designed to be common in many different OpenEmbedded derived systems (including Yocto projects).

3. Poky: a reference distribution.
Poky is the name of the reference distribution or reference OS of Yocto project. Poky includes the OpenEmbedded Build System (BitBake and OpenEmbedded-Core) and a set of metadata to help you start building your own distribution. Poky uses OpenEmbedded Build System to build a small embedded operating system. Poky is an integration layer on top of OE-Core. Poky provides the following:

- A basic level of distro infrastructure to illustrate how to customize the distro.
- A means to verify the Yocto Project components.

![Fig.3](https://www.seeedstudio.com/blog/wp-content/uploads/2021/09/image-21.png)

## The OpenEmbedded Build System Workflow

When downloading the build system, the Poky build ‘file’ is called a recipe and a layer. You can modify, assign, or any way you need to create your own customized embedded Linux.

The following diagram represents the high-level workflow of a build. This section expands to the fundamental input, output, process, and metadata logical blocks that make up the workflow.

![Fig.4](https://www.yoctoproject.org/docs/2.7/overview-manual/figures/YP-flow-diagram.png)

We also tested hardware and interfaces usage for Yocto Image we created. Please click here if you want to verify the hardware availability with different images to expand your ideas.

Yocto Project is suitable for embedded and IoT devices with limited resources: Unlike a full Linux distribution, embedded developers can use Yocto Project to create a condensed version for embedded devices. You only need to add the necessary functional tools and software packages into your Linux System. If the device has a display part, you can use system components such as X11, GTK+, Qt, Clutter, and SDL to create a Distro with a better UI experience.

## Yocto Project Terminology

Before we get started with Yocto Project in real, let’s take a look and keep in review the following terminology to help you understand the structure and all components of the Yocto Project better.

- Yocto Project: An open-source community project that allows you to build your own Linux system.
- OpenEmbedded: The build system specific to the Yocto Project.
- OpenEmbedded-Core: Metadata composed of basic recipes, classes, and related files. These metadata are designed to be common in many different OpenEmbedded derived systems (including Yocto projects).
- BitBake: The core tool of the OpenEmbedded build system, responsible for readinging metadata, generating task lists from it, and then performing these tasks.
- Poky: Includes the OpenEmbedded Build System (BitBake and OpenEmbedded-Core) and a set of metadata to help you start building your own distribution. Poky uses OpenEmbedded Build System to build a small embedded operating system. Poky is an integration layer on top of OE-Core.
- Recipe: The most common form of metadata. Recipe contains a list of settings and tasks used to build a binary image file. Recipe describes where you get the code and which (code) patch you need to apply. At the same time, Recipe also describes the dependencies on other recipes or libraries, as well as configuration and compilation options. Recipe is stored in Layer.
- Layer: A collection of related recipes. A Layer is a repository contains relevant metadata that tells the OpenEmbedded build system how to build the target. Yocto Project’s layer model promotes collaboration, sharing, customization, and reuse in the Yocto Project development environment. Layers logically separate the information of your project.
- Metadata: Layer contains the recipe files, patches, and additional files provided by the user, other information referring to the build instructions, and data that controls what and how to build. A good example of the software layer might be the meta-Qt5 Layer from the OpenEmbedded Layer Index.
