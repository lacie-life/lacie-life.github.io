---
title: Setting Up Environment for Building Boot to Qt (Raspberri Pi 4)
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-01-12 11:11:11 +0700
categories: [Linux, Qt]
tags: [tutorial]
render_with_liquid: false
---

# Setting Up Environment for Building Boot to Qt (Raspberri Pi 4)

## Google repo tools install

```
mkdir -p ~/.bin
PATH="${HOME}/.bin:${PATH}"
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.bin/repo
chmod a+rx ~/.bin/repo
```

## Initializing Yocto Build Environment

### Get manifest file

[Link](https://code.qt.io/cgit/yocto/boot2qt-manifest.git/tree/)

```
cd <BuildDir>
repo init -u git://code.qt.io/yocto/boot2qt-manifest -m <manifest>
repo sync
```
## Building the Image and Toolchain

```
export MACHINE=raspberrypi4
source ./setup-environment.sh

bitbake b2qt-embedded-qt5-image
bitbake meta-toolchain-b2qt-embedded-qt5-sdk
```

