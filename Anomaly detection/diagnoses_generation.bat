:: xcopy path\to\dirs destination\path /E

set nreps=5
set test_split=0.5
set validation_split=0.2


del /F /Q Input\EventLogs\*
for /D %%p IN ("Diagnoses\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

copy TestResults\* Input\EventLogs

python diagnoses_generation.py %nreps% %validation_split% %test_split%
