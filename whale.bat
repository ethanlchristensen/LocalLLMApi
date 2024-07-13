@echo off

echo Stopping and removing old containers
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=localllmapi"') do docker stop %%i
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=localllmapi"') do docker rm %%i

echo Y | docker system prune -a

echo Building the container
docker build -t localllmapi .

echo Running the container in detached mode
docker run -d -p 8000:8000 --memory=2048m --name localllmapi localllmapi