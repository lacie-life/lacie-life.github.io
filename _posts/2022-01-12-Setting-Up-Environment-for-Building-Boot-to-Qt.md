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
```

