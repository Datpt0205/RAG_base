#!/bin/sh
ollama pull deepseek-r1:14b
exec ollama serve
chmod +x init-ollama.sh