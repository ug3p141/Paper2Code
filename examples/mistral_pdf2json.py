import os
import json
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2412.14042"
    },
    include_image_base64=True
)

response_dict = json.loads(ocr_response.model_dump_json())

# Save the complete response as JSON
with open("document_ocr.json", "w", encoding="utf-8") as f:
    json.dump(response_dict, f, indent=4)
