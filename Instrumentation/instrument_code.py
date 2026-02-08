import sys
import os
from google import genai
from google.genai import types
from openai import OpenAI

input_folder = "Input/"
output_folder = "Output/"

def read_input_files():

	instrumentation_prompt_path = os.path.join(input_prompts_folder, "instrumentation_prompt.txt")
	source_path = os.path.join(input_folder, "start_of_mission.py")
	test_path = os.path.join(input_folder, "test_suite.py")
	bpmn_path = os.path.join(input_folder, "START_OF_MISSION.bpmn")

	if not os.path.exists(instrumentation_prompt_path):
		raise FileNotFoundError(f"Missing file: {instrumentation_prompt_path}")
	if not os.path.exists(source_path):
		raise FileNotFoundError(f"Missing file: {source_path}")
	if not os.path.exists(bpmn_path):
		raise FileNotFoundError(f"Missing file: {bpmn_path}")
	if not os.path.exists(test_path):
		raise FileNotFoundError(f"Missing file: {test_path}")	

	with open(instrumentation_prompt_path, "r", encoding="utf-8") as f:
		instrumentation_prompt = f.read().strip()
	with open(source_path, "r", encoding="utf-8") as f:
		source_code = f.read().strip()
	with open(bpmn_path, "r", encoding="utf-8") as f:
		bpmn_model = f.read().strip()
	with open(bpmn_path, "r", encoding="utf-8") as f:
		test_suite = f.read().strip()	

	return instrumentation_prompt, source_code, bpmn_model, test_suite

def generate_instrumented_code(instrumentation_prompt, source_code, model, bpmn_model, test_suite):

	instrumentation_response = None
	
	if model == "gemini-3.0-pro":
		client = genai.Client(api_key="")
		instrumentation_response = client.models.generate_content(
				model="gemini-3.0-pro",
				config=types.GenerateContentConfig(system_instruction=instrumentation_prompt + "\n" + bpmn_model + "\n" + test_suite),
				contents=source_code
			)
			
		
	elif model == "gemini-3.0-flash":
		client = genai.Client(api_key="")
		instrumentation_response = client.models.generate_content(
			model="gemini-3.0-flash",
			config=types.GenerateContentConfig(system_instruction=instrumentation_prompt + "\n" + bpmn_model + "\n" + test_suite),
			contents=source_code
		)
		
		
	elif model == "claude-sonnet-4.5":
		client = OpenAI(
		  base_url="https://openrouter.ai/api/v1",
		  api_key="",
		)
		completion = client.chat.completions.create(
			model="anthropic/claude-sonnet-4.5",
			messages=[
				{
					"role": "system",
					"content": instrumentation_prompt + "\n" + bpmn_model + "\n" + test_suite
				},
				{
					"role": "user",
					"content": source_code
				}
			],
		)
		
	elif model == "claude-haiku-4.5":
		client = OpenAI(
		  base_url="https://openrouter.ai/api/v1",
		  api_key="",
		)
		completion = client.chat.completions.create(
			model="anthropic/claude-haiku-4.5",
			messages=[
				{
					"role": "system",
					"content": instrumentation_prompt + "\n" + bpmn_model + "\n" + test_suite
				},
				{
					"role": "user",
					"content": source_code
				}
			],
		)			
	

	return instrumentation_response
	
def write_output(instrumentation_response, filename):

	output_text = instrumentation_response.text
	output_text = output_text.replace("```python", "").replace("```", "").strip()	

	with open(output_folder + filename, "w", encoding="utf-8") as f:
		f.write(output_text)

	return None

try:
	model = sys.argv[1]

except:
	print("Enter the right number of input arguments.")
	sys.exit()

instrumentation_prompt, source_code, bpmn_model, test_suite = read_input_files()
instrumentation_response = generate_instrumented_code(instrumentation_prompt, source_code, model, bpmn_model, test_suite)
write_output(instrumentation_response, "start_of_mission.py")
	