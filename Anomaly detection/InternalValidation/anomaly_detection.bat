set ad_method=ft dbscan ae
set anomaly_type=UA WOA MA ALL
set model=claude-sonnet-4.5
set n_reps=0

for /D %%p IN ("InternalValidationResults\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /l %%a in (0, 1, %n_reps%) do (

	mkdir InternalValidationResults\%%a
	
	for %%y in (%ad_method%) do (
	
		mkdir InternalValidationResults\%%a\%%y
	
		for %%z in (%anomaly_type%) do (
		
			del /F /Q Output\Metrics\*
		
			python anomaly_detection.py %%y %%z %model% %%a
			
			copy Output\Metrics\Metrics.txt InternalValidationResults\%%a\%%y
			copy Output\Metrics\Parameter_set.txt InternalValidationResults\%%a\%%y
			
			ren InternalValidationResults\%%a\%%y\Metrics.txt Metrics_%%z.txt
			ren InternalValidationResults\%%a\%%y\Parameter_set.txt Parameter_set_%%z.txt
			
		)
	)
)





