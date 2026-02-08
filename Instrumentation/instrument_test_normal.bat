:: Options:
:: models=[gemini-2.5-pro, gemini-2.5-flash]
:: instrumentation_type=[dk_instrumentation, free_instrumentation, procedure_instrumentation]

set models=gemini-2.5-pro

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%m in (%models%) do (

	mkdir Results\%%m
	
	del /F /Q Output\IC\*
	
	python instrument_code.py %%m
	
	copy Output\IC\* Results\%%m
	

)


