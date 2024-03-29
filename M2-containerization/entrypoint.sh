#!/bin/bash
umask 002
mkdir -p /app/models /app/results
chmod 775 /app/models /app/results
exec "$@"