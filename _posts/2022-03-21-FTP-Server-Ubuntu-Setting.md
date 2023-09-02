---
title: Setup FTP Server with VSFTPD on Ubuntu 20.04
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-03-21 11:11:11 +0700
categories: [Skill]
tags: [tutorial]
# img_path: /assets/img/post_assest/
render_with_liquid: false
---

FTP (File Transfer Protocol) is a standard network protocol used to transfer files to and from a remote network. There are several open-source FTP servers available for Linux. The most known and widely used are PureFTPd , ProFTPD , and vsftpd . We’ll be installing vsftpd (Very Secure Ftp Daemon), a stable, secure, and fast FTP server. We will also show you how to configure the server to restrict users to their home directory and encrypt the entire transmission with SSL/TLS.

Although FTP is a very popular protocol, for more secure and faster data transfers, you should use SCP or SFTP .

## Installing vsftpd on Ubuntu 20.04

The vsftpd package is available in the Ubuntu repositories. To install it, execute the following commands:

```
sudo apt update
sudo apt install vsftpd
```

The ftp service will automatically start once the installation process is complete. To verify it, print the service status:

```
sudo systemctl status vsftpd
```

The output should show that the vsftpd service is active and running:

## Configuring vsftpd

The vsftpd server configuration is stored in the /etc/vsftpd.conf file.

Most of the server settings are well documented inside the file. For all available options, visit the vsftpd documentation page.

In the following sections, we will go over some important settings needed to configure a secure vsftpd installation.

Start by opening the vsftpd configuration file:

```
sudo nano /etc/vsftpd.conf
```

### 1. FTP access 

We’ll allow access to the FTP server only to the local users. Search for the anonymous_enable and local_enable directives in /etc/vsftpd.conf and verify your configuration match to lines below:

```
anonymous_enable=NO
local_enable=YES
```

### 2. Enabling uploads

Locate and uncomment the write_enable directive to allow filesystem changes, such as uploading and removing files:

```
write_enable=YES
```

### 3. Chroot jail 

To prevent local FTP users to access files outside of their home directories, uncomment the lne starting with chroot_local_user:

```
chroot_local_user=YES
```

By default, for security reasons, when chroot is enabled, vsftpd will refuse to upload files if the directory that the users are locked in is writable.

Use one of the solutions below to allow uploads when chroot is enabled:

- Method 1. - The recommended option is to keep the chroot feature enabled and configure FTP directories. In this example, we will create an ftp directory inside the user home, which will serve as the chroot and a writable uploads directory for uploading files:

```
user_sub_token=$USER
local_root=/home/$USER/ftp
```

- Method 2. - Another option is to enable the allow_writeable_chroot directive:

```
allow_writeable_chroot=YES
```

Use this option only if you must grant writable access to your user to its home directory.

### 4. Passive FTP Connections

By default, vsftpd uses active mode. To use passive mode, set the minimum and maximum range of ports:

```
pasv_min_port=30000
pasv_max_port=31000
```

You can use any port for passive FTP connections. When the passive mode is enabled, the FTP client opens a connection to the server on a random port in the range you have chosen.

### 5. Limiting User Login

You can configure vsftpd to permit only certain users to log in. To do so, add the following lines at the end of the file:

```
userlist_enable=YES
userlist_file=/etc/vsftpd.user_list
userlist_deny=NO
```

When this option is enabled, you need to explicitly specify which users can log in by adding the user names to the /etc/vsftpd.user_list file (one user per line).

### 6. Securing Transmissions with SSL/TLS


To encrypt the FTP transmissions with SSL/TLS, you’ll need to have an SSL certificate and configure the FTP server to use it.

You can use an existing SSL certificate signed by a trusted Certificate Authority or create a self-signed certificate.
If you have a domain or subdomain pointing to the FTP server’s IP address, you can quickly generate a free Let’s Encrypt SSL certificate.
We will generate a 2048-bit private key and self-signed SSL certificate that will be valid for ten years:

```
sudo openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout /etc/ssl/private/vsftpd.pem -out /etc/ssl/private/vsftpd.pem
```

Both the private key and the certificate will be saved in the same file.

Once the SSL certificate is created open the vsftpd configuration file:

```
sudo nano /etc/vsftpd.conf
```

Find the rsa_cert_file and rsa_private_key_file directives, change their values to the pam file path and set the ssl_enable directive to YES:

```
rsa_cert_file=/etc/ssl/private/vsftpd.pem
rsa_private_key_file=/etc/ssl/private/vsftpd.pem
ssl_enable=YES
```

If not specified otherwise, the FTP server will use only TLS to make secure connections.

### Restart the vsftpd Service

Once you are done editing, the vsftpd configuration file (excluding comments) should look something like this:

```
listen=NO
listen_ipv6=YES
anonymous_enable=NO
local_enable=YES
write_enable=YES
dirmessage_enable=YES
use_localtime=YES
xferlog_enable=YES
connect_from_port_20=YES
chroot_local_user=YES
secure_chroot_dir=/var/run/vsftpd/empty
pam_service_name=vsftpd
rsa_cert_file=/etc/ssl/private/vsftpd.pem
rsa_private_key_file=/etc/ssl/private/vsftpd.pem
ssl_enable=YES
user_sub_token=$USER
local_root=/home/$USER/ftp
pasv_min_port=30000
pasv_max_port=31000
userlist_enable=YES
userlist_file=/etc/vsftpd.user_list
userlist_deny=NO
```

Save the file and restart the vsftpd service for changes to take effect:

```
sudo systemctl restart vsftpd
```

## Opening the Firewall 

If you are running a UFW firewall , you’ll need to allow FTP traffic.

To open port 21 (FTP command port), port 20 (FTP data port), and 30000-31000 (Passive ports range), run the following commands:

```
sudo ufw allow 20:21/tcp
sudo ufw allow 30000:31000/tcp
```

To avoid being locked out, make sure port 22 is open:

```
sudo ufw allow OpenSSH
```

Reload the UFW rules by disabling and re-enabling UFW:

```
sudo ufw disable
sudo ufw enable
```

To verify the changes run:


```
sudo ufw status
```

Output

```
Status: active

To                         Action      From
--                         ------      ----
20:21/tcp                  ALLOW       Anywhere
30000:31000/tcp            ALLOW       Anywhere
OpenSSH                    ALLOW       Anywhere
20:21/tcp (v6)             ALLOW       Anywhere (v6)
30000:31000/tcp (v6)       ALLOW       Anywhere (v6)
OpenSSH (v6)               ALLOW       Anywhere (v6)
```

## Creating FTP User

To test the FTP server, we will create a new user.

- If the user to which you want to grant FTP access already exists, skip the 1st step.
- If you set allow_writeable_chroot=YES in your configuration file, skip the 3rd step.

### 1. Create a new user named newftpuser:

```
sudo adduser newftpuser
```

### 2. Add the user to the allowed FTP users list:

```
echo "newftpuser" | sudo tee -a /etc/vsftpd.user_list
```

### 3. Create the FTP directory tree and set the correct permissions :

```
sudo mkdir -p /home/newftpuser/ftp/upload
sudo chmod 550 /home/newftpuser/ftp
sudo chmod 750 /home/newftpuser/ftp/upload
sudo chown -R newftpuser: /home/newftpuser/ftp
```

As discussed in the previous section, the user will be able to upload its files to the ftp/upload directory.

At this point, your FTP server is fully functional. You should be able to connect to the server using any FTP client that can be configured to use TLS encryption, such as FileZilla .

## Disabling Shell Access

By default, when creating a user, if not explicitly specified, the user will have SSH access to the server. To disable shell access, create a new shell that will print a message telling the user that their account is limited to FTP access only.

Run the following commands to create the /bin/ftponly file and make it executable:

```
echo -e '#!/bin/sh\necho "This account is limited to FTP access only."' | sudo tee -a  /bin/ftponly
sudo chmod a+x /bin/ftponly
```

Append the new shell to the list of valid shells in the /etc/shells file:

```
echo "/bin/ftponly" | sudo tee -a /etc/shells
```

Change the user shell to /bin/ftponly:

```
sudo usermod newftpuser -s /bin/ftponly
```

You can use the same command to change the shell of all users you want to give only FTP access.

## Upload file to FTP server


```
import ftplib
ftp = ftplib.FTP("192.168.2.222")
ftp.login("test", "123qwe")
localfile='https://d1ny9casiyy5u5.cloudfront.net/tmp/test.txt'
remotefile='test.txt'
with open(localfile, "rb") as file:
    ftp.storbinary('STOR %s' % remotefile, file)
```



