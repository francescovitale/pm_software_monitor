# Requirements to run the method

## Packages
This project has been executed on a Windows 11 machine with Pytest 8.4.2 and Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.11.2
- scikit-learn 1.3.0
- google-genai 1.45.0
- openai 2.7.2

Please note that the list above is not comprehensive and there could be other requirements for running the project.

# Project structure

```
pm_software_monitor/
├── Anomaly detection/  # Fault injection, diagnoses generation, and control-flow anomaly detection code
├── Instrumentation/    # LLM-based source-code instrumentation code and test suite application to the instrumented code
├── Results/            # The results of LLM-based source-code instrumentation, test suite application, and control-flow anomaly detection
├── SoM code and test suite/  # The source code of the SoM prototype and the related test suite
```

## SoM code and test suite

This part of the project contains the Start of Mission (SoM) prototype source code. This prototype implements and simulates the essentials of the procedure described in the ERTMS/ETCS standard SUBSET-026 (https://www.era.europa.eu/era-folder/1-ccs-tsi-appendix-mandatory-specifications-etcs-b4-r1-rmr-gsm-r-b1-mr1-frmcs-b0-ato-b1). 

In addition, this part contains the test suite of the prototype, which includes 50 successful tests. This test suite is used to stimulate the prototype and collect the logs during the execution of the code instrumented by the LLMs.

## Instrumentation

This part of the project contains the code that interacts with the four LLMs through their APIs (instrument_code.py). The script reads the SoM code and test suite, the instrumentation prompt, and the process model. 

The Test subfolder contains the script to execute the test suite against the instrumented source codes obtained through the LLMs. Subsequently, another script is used to extract the overall XES event logs from the different log files collected from each individual test case. The DOS script test.bat combines the execution of the test suite and the event log extraction script automatically.

## Anomaly detection

This part of the project contains the logic to extract diagnoses from the XES event logs (diagnoses_generation.bat). 

The diagnoses can be used for internal validation of the logs of a single LLM (InternalValidation) and the inter-LLM validation (InterLLMValidation). The former is used to evaluate the control-flow anomaly detection performance of an individual LLM, whereas the latter is used to check how different instrumentations can impact the control-flow anomaly detection performance.

## Results

This part of the project collects the results from Instrumentation and Anomaly detection. In particular:

- InstrumentationResults collects the instrumented versions of the SoM prototype obtained by the four different LLMs we used in our experiments.
- TestResults collects the event logs resulting from the execution of the test suites on the instrumented version of the SoM prototypes.
- InternalValidationResults collects the control-flow anomaly detection performance results of one of the LLMs.
- InterLLMValidationResults collects the control-flow anomaly detection performance results obtained by classifying the traces of the less-performing LLMs using the monitor learned from the best-perfroming LLM.


