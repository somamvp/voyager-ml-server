#!/bin/bash
ssh -i somaMVP.pem -R 5001:127.0.0.1:8000 ec2-user@43.200.197.251
