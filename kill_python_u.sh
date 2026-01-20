#!/usr/bin/env bash

# This script kills all processes that were started with a command string beginning with 'python -u'.
# It targets the full command line (-f) and uses a regular expression to match the start.

echo "Searching for processes starting with 'python -u'..."

# Get PIDs of processes matching the pattern
# The pattern targets 'python -u' at the start of the command line string as seen in ps
PIDS=$(ps -eo pid,command | grep "python -u" | grep -v "grep" | awk '{print $1}')

if [ -z "$PIDS" ]; then
    echo "No processes found starting with 'python -u'."
else
    echo "Killing the following PIDs:"
    echo "$PIDS"
    echo "$PIDS" | xargs kill -9
    echo "Done."
fi
