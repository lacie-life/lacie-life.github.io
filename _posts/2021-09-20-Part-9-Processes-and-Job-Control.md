---
title: Process and Job Control
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2021-09-20 11:11:11 +0700
categories: [Linux]
tags: [Tutorial]
render_with_liquid: false
---

### Process and Job Control

1. Most Important Commands:

ps -e - Display all Processes

ps -ef - Display all Processes , full

ps -u <user> - Display user's processes

top - Interactive Process viewer (as Task Manager in Windows)

htop - Interactive Process viewer (as Task Manager in Windows)

2. Background and Foreground Processes:

command & - Start command in the background

Ctrl - c - kill the foreground Process

Ctrl - d - Exit the foreground Process

Ctrl - z - Suspend the foreground Process

bg <%num> - Background Suspended Process

fg <%num> - Foreground Suspended Process

Kill <PID / Name> - Kill Process by PID or Name

Kill -l - Display a list of signals

jobs <%num> - List Jobs

3. Scheduling Repeated Jobs with Cron

crontab <file> - Install a new crontab from a file

crontab -l - List your cron jobs

crontab -e - Edit your cron jobs

crontab -r - Remove all your cron jobs



