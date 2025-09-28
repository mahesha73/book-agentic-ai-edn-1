# Sample_JsonOutputParser.py
# Section 4.3.1
# Page 101

from langchain.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
output = llm_chain.run(input)
parsed_output = json_parser.parse(output)
