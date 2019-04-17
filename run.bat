@echo off

IF %1.==. (
	set arg=1
) ELSE (
	set arg=%1  
)

::Not Implemented
::echo --------------------------------------Quora-------------------------------------
::cd quora
::python main.py %arg%
::cd ..

echo -------------------------------------Reddit-------------------------------------
cd reddit
python main.py %arg%
cd ..

echo ----------------------------------StackExchange---------------------------------
cd stackExchange
python main.py %arg%
cd..

::Not Implemented
::echo --------------------------------------Yahoo-------------------------------------
::cd yahoo
::python main.py %arg%
::cd..

@pause