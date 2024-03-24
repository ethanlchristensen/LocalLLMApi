@echo off

echo Building that bizzy
docker build -t localllmapi .
echo Running that bizzy
docker run -p 8000:8000 localllmapi