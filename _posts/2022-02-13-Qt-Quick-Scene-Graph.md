---
title: Qt Framework - Qt Quick Scene Graph
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-02-13 11:11:11 +0700
categories: [Skill, Software, Qt]
tags: [tutorial]
img_path: /assets/img/post_assest/
render_with_liquid: false
---

## The Scene Graph in Qt Quick

Qt Quick 2 makes use of a dedicated scene graph that is then traversed and rendered via a graphics API such as OpenGL ES, OpenGL, Vulkan, Metal, or Direct 3D. Using a scene graph for graphics rather than the traditional imperative painting systems (QPainter and similar), means the scene to be rendered can be retained between frames and the complete set of primitives to render is known before rendering starts. This opens up for a number of optimizations, such as batch rendering to minimize state changes and discarding obscured primitives.

For example, say a user-interface contains a list of ten items where each item has a background color, an icon and a text. Using the traditional drawing techniques, this would result in 30 draw calls and a similar amount of state changes. A scene graph, on the other hand, could reorganize the primitives to render such that all backgrounds are drawn in one call, then all icons, then all the text, reducing the total amount of draw calls to only 3. Batching and state change reduction like this can greatly improve performance on some hardware.

The scene graph is closely tied to Qt Quick 2.0 and can not be used stand-alone. The scene graph is managed and rendered by the <b> QQuickWindow </b> class and custom Item types can add their graphical primitives into the scene graph through a call to <b> QQuickItem::updatePaintNode() </b>.

The scene graph is a graphical representation of the Item scene, an independent structure that contains enough information to render all the items. Once it has been set up, it can be manipulated and rendered independently of the state of the items. On many platforms, the scene graph will even be rendered on a dedicated render thread while the GUI thread is preparing the next frame's state.

## Qt Quick Scene Graph Structure

The scene graph is composed of a number of predefined node types, each serving a dedicated purpose. Although we refer to it as a scene graph, a more precise definition is node tree. The tree is built from <b> QQuickItem </b> types in the QML scene and internally the scene is then processed by a renderer which draws the scene. The nodes themselves do not contain any active drawing code nor virtual <b> paint() </b> function.

Even though the node tree is mostly built internally by the existing Qt Quick QML types, it is possible for users to also add complete subtrees with their own content, including subtrees that represent 3D models.

### Nodes

The most important node for users is the <b> QSGGeometryNode </b>. It is used to define custom graphics by defining its geometry and material. The geometry is defined using <b> QSGGeometry </b> and describes the shape or mesh of the graphical primitive. It can be a line, a rectangle, a polygon, many disconnected rectangles, or complex 3D mesh. The material defines how the pixels in this shape are filled.

A node can have any number of children and geometry nodes will be rendered so they appear in child-order with parents behind their children.

The available nodes are:
|Type|Description|
|----|------------|
|QSGClipNode |Implements the clipping functionality in the scene graph|
|QSGGeometryNode |Used for all rendered content in the scene graph|
|QSGNode |The base class for all nodes in the scene graph|
|QSGOpacityNode |Used to change opacity of nodes|
|QSGTransformNode |Implements transformations in the scene graph|

Custom nodes are added to the scene graph by subclassing  <b> QQuickItem::updatePaintNode() </b> and setting the  <b> QQuickItem::ItemHasContents </b> flag.

#### Preprocessing

Nodes have a virtual <b> QSGNode::preprocess() </b> function, which will be called before the scene graph is rendered. Node subclasses can set the flag  <b> QSGNode::UsePreprocess </b> and override the  <b> QSGNode::preprocess() </b> function to do final preparation of their node. For example, dividing a bezier curve into the correct level of detail for the current scale factor or updating a section of a texture.

#### Node Ownership

Ownership of the nodes is either done explicitly by the creator or by the scene graph by setting the flag  <b> QSGNode::OwnedByParent </b>. Assigning ownership to the scene graph is often preferable as it simplifies cleanup when the scene graph lives outside the GUI thread.

### Materials

The material describes how the interior of a geometry in a <b> QSGGeometryNode </b> is filled. It encapsulates graphics shaders for the vertex and fragment stages of the graphics pipeline and provides ample flexibility in what can be achieved, though most of the Qt Quick items themselves only use very basic materials, such as solid color and texture fills.

For users who just want to apply custom shading to a QML Item type, it is possible to do this directly in QML using the <b> ShaderEffect </b> type.

Below is a complete list of material classes:

| Type | Description |
|------|-------------|
|QSGFlatColorMaterial |Convenient way of rendering solid colored geometry in the scene graph|
|QSGMaterial |Encapsulates rendering state for a shader program|
|QSGMaterialRhiShader |Represents a graphics API independent shader program|
|QSGMaterialShader |Represents an OpenGL shader program in the renderer|
|QSGMaterialType |Used as a unique type token in combination with QSGMaterial|
|QSGOpaqueTextureMaterial |Convenient way of rendering textured geometry in the scene graph|
|QSGTextureMaterial |Convenient way of rendering textured geometry in the scene graph|
|QSGVertexColorMaterial |Convenient way of rendering per-vertex colored geometry in the scene graph|

### Convenience Nodes

The scene graph API is low-level and focuses on performance rather than convenience. Writing custom geometries and materials from scratch, even the most basic ones, requires a non-trivial amount of code. For this reason, the API includes a few convenience classes to make the most common custom nodes readily available.

- QSGSimpleRectNode - a QSGGeometryNode subclass which defines a rectangular geometry with a solid color material.

- QSGSimpleTextureNode - a QSGGeometryNode subclass which defines a rectangular geometry with a texture material.

## Scene Graph and Rendering

The rendering of the scene graph happens internally in the ?<b> QQuickWindow </b> class, and there is no public API to access it. There are, however, a few places in the rendering pipeline where the user can attach application code. This can be used to add custom scene graph content or to insert arbitrary rendering commands by directly calling the graphics API (OpenGL, Vulkan, Metal, etc.) that is in use by the scene graph. The integration points are defined by the render loop.

There are three render loop variants available: basic, windows, and threaded. Out of these, basic and windows are single-threaded, while threaded performs scene graph rendering on a dedicated thread. Qt attempts to choose a suitable loop based on the platform and possibly the graphics drivers in use. When this is not satisfactory, or for testing purposes, the environment variable QSG_RENDER_LOOP can be used to force the usage of a given loop. To verify which render loop is in use, enable the qt.scenegraph.general logging category.

### Threaded Render Loop ("threaded")

On many configurations, the scene graph rendering will happen on a dedicated render thread. This is done to increase parallelism of multi-core processors and make better use of stall times such as waiting for a blocking swap buffer call. This offers significant performance improvements, but imposes certain restrictions on where and when interaction with the scene graph can happen.

The following is a simple outline of how a frame gets rendered with the threaded render loop and OpenGL. The steps are the same with other graphics APIs as well, apart from the OpenGL context specifics.

![Fig.1](https://doc.qt.io/qt-5/images/sg-renderloop-threaded.png)

1. A change occurs in the QML scene, causing <b> QQuickItem::update() </b> to be called. This can be the result of for instance an animation or user input. An event is posted to the render thread to initiate a new frame.

2. The render thread prepares to draw a new frame and initiates a block on the GUI thread.

3. While the render thread is preparing the new frame, the GUI thread calls <b> QQuickItem::updatePolish() </b> to do final touch-up of items before they are rendered.

4. GUI thread is blocked.

5. The <b> QQuickWindow::beforeSynchronizing() </b> signal is emitted. Applications can make direct connections (using <b> Qt::DirectConnection <b>) to this signal to do any preparation required before calls to <b> QQuickItem::updatePaintNode() </b>.

6. Synchronization of the QML state into the scene graph. This is done by calling the <b> QQuickItem::updatePaintNode() </b> function on all items that have changed since the previous frame. This is the only time the QML items and the nodes in the scene graph interact.

7. GUI thread block is released.

8. The scene graph is rendered:

    - The <b> QQuickWindow::beforeRendering() </b> signal is emitted. Applications can make direct connections (using <b> Qt::DirectConnection </b>) to this signal to use custom graphics API calls which will then stack visually beneath the QML scene.

    - Items that have specified <b> QSGNode::UsePreprocess </b>, will have their <b> QSGNode::preprocess() </b> function invoked.

    - The renderer processes the nodes.

    - The renderer generates states and records draw calls for the graphics API in use.

    - The <b> QQuickWindow::afterRendering() </b> signal is emitted. Applications can make direct connections (using <b> Qt::DirectConnection </b>) to this signal to issue custom graphics API calls which will then stack visually over the QML scene.

    - The frame is now ready. The buffers are swapped (OpenGL), or a present command is recorded and the command buffers are submitted to a graphics queue (Vulkan, Metal). <b> QQuickWindow::frameSwapped() </b> is emitted.

9. While the render thread is rendering, the GUI is free to advance animations, process events, etc.

The threaded renderer is currently used by default on Windows with opengl32.dll, Linux excluding Mesa llvmpipe, macOS with Metal, mobile platforms, and Embedded Linux with EGLFS, and with Vulkan regardless of the platform, but this is subject to change. It is always possible to force use of the threaded renderer by setting QSG_RENDER_LOOP=threaded in the environment.

### Non-threaded Render Loops ("basic" and "windows")

The non-threaded render loop is currently used by default on Windows with ANGLE or a non-default opengl32 implementation, macOS with OpenGL, and Linux with some drivers. For the latter this is mostly a precautionary measure, as not all combinations of OpenGL drivers and windowing systems have been tested. At the same time implementations like ANGLE or Mesa llvmpipe are not able to function properly with threaded rendering at all so not using threaded rendering is essential for these.

On macOS and OpenGL, the threaded render loop is not supported when building with XCode 10 (10.14 SDK) or later, since this opts in to layer-backed views on macOS 10.14. You can build with Xcode 9 (10.13 SDK) to opt out of layer-backing, in which case the threaded render loop is available and used by default. There is no such restriction with Metal.

By default windows is used for non-threaded rendering on Windows with ANGLE, while basic is used for all other platforms when non-threaded rendering is needed.

Even when using the non-threaded render loop, you should write your code as if you are using the threaded renderer, as failing to do so will make the code non-portable.

The following is a simplified illustration of the frame rendering sequence in the non-threaded renderer.

![Fig.2](https://doc.qt.io/qt-5/images/sg-renderloop-singlethreaded.png)

#### Custom control over rendering with QQuickRenderControl

When using <b> QQuickRenderControl </b>, the responsibility for driving the rendering loop is transferred to the application. In this case no built-in render loop is used. Instead, it is up to the application to invoke the polish, synchronize and rendering steps at the appropriate time. It is possible to implement either a threaded or non-threaded behavior similar to the ones shown above.

#### Mixing Scene Graph and the native graphics API

The scene graph offers two methods for integrating application-provided graphics commands: by issuing OpenGL, Vulkan, Metal, etc. commands directly, and by creating a textured node in the scene graph.

By connecting to the <b> QQuickWindow::beforeRendering() </b> and <b> QQuickWindow::afterRendering() </b> signals, applications can make OpenGL calls directly into the same context as the scene graph is rendering to. With APIs like Vulkan or Metal, applications can query native objects, such as, the scene graph's command buffer, via QSGRendererInterface, and record commands to it as they see fit. As the signal names indicate, the user can then render content either under a Qt Quick scene or over it. The benefit of integrating in this manner is that no extra framebuffer nor memory is needed to perform the rendering, and a possibly expensive texturing step is eliminated. The downside is that Qt Quick decides when to call the signals and this is the only time the OpenGL application is allowed to draw.

#### Custom Items using QPainter

The <b> QQuickItem </b> provides a subclass, <b> QQuickPaintedItem </b>, which allows the users to render content using <b> QPainter </b>.

### Logging Support

The scene graph has support for a number of logging categories. These can be useful in tracking down both performance issues and bugs in addition to being helpful to Qt contributors.

- qt.scenegraph.time.texture - logs the time spent doing texture uploads
- qt.scenegraph.time.compilation - logs the time spent doing shader compilation
- qt.scenegraph.time.renderer - logs the time spent in the various steps of the renderer
- qt.scenegraph.time.renderloop - logs the time spent in the various steps of the render loop
- qt.scenegraph.time.glyph - logs the time spent preparing distance field glyphs
- qt.scenegraph.general - logs general information about various parts of the scene graph and the graphics stack
- qt.scenegraph.renderloop - creates a detailed log of the various stages involved in rendering. This log mode is primarily useful for developers working on Qt.

The legacy QSG_INFO environment variable is also available. Setting it to a non-zero value enables the qt.scenegraph.general category.

## Scene Graph Backend

In addition to the public API, the scene graph has an adaptation layer which opens up the implementation to do hardware specific adaptations. This is an undocumented, internal and private plugin API, which lets hardware adaptation teams make the most of their hardware. It includes:

- Custom textures; specifically the implementation of <b>  QQuickWindow::createTextureFromImage </b> and the internal representation of the texture used by <b> Image </b> and <b> BorderImage </b> types.
- Custom renderer; the adaptation layer lets the plugin decide how the scene graph is traversed and rendered, making it possible to optimize the rendering algorithm for a specific hardware or to make use of extensions which improve performance.
- Custom scene graph implementation of many of the default QML types, including its text and font rendering.
- Custom animation driver; allows the animation system to hook into the low-level display vertical refresh to get smooth rendering.
- Custom render loop; allows better control over how QML deals with multiple windows.


