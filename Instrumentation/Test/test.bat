:: set models=gemini-3.0-flash gemini-3.0-pro claude-sonnet-4.5 claude-haiku-4.5
set models=gemini-3.0-flash gemini-3.0-pro claude-sonnet-4.5 claude-haiku-4.5
set nreps=1 2 3 4 5


for /D %%p IN ("TestResults\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)




for %%r in (%nreps%) do (

	mkdir TestResults\%%r

	for %%m in (%models%) do (

		
		del /F /Q Input\TS\*
		del /F /Q Output\TS\*
		del /F /Q Input\ELE\*
		del /F /Q Output\ELE\*
		
		copy InstrumentationResults\%%r\%%m\start_of_mission.py Input\TS
		
		pytest test_suite.py
		
		call clean_test_cache

		copy Output\TS\* Input\ELE
		
		python event_log_extraction.py
		
		copy Output\ELE\* TestResults\%%r
		ren TestResults\%%r\software_log.xes %%m.xes

	)

)




