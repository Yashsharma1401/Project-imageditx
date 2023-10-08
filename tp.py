import requests

# Specify the API endpoint and API key
api_endpoint = 'https://api.ocr.space/parse/image'
api_key = 'K85000163888957'  # Replace with your own API key

# Open the image file
image_path = 'images.jpg'
image_path = 'images1.png'
# image_path = 'images2.png'
image_data = open(image_path, 'rb')

# Prepare the request payload
payload = {
    'apikey': api_key,
    'language': 'eng',  # Specify the language of the text in the image
}

# Send the POST request to the API
response = requests.post(api_endpoint, files={'image': image_data}, data=payload)
response_data = response.json()

# Extract the text from the response
if response_data['IsErroredOnProcessing']:
    print('Error occurred during text extraction.')
else:
    extracted_text = response_data['ParsedResults'][0]['ParsedText']
    print(extracted_text)
