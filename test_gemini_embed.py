from openai import OpenAI
import os

client = OpenAI(
    api_key="AIzaSyACwPiff5223i8uTpdDT1lYAXuYzniUYOk",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.embeddings.create(
    input=["berapa budgetnya itu?"],
    model="text-embedding-004"
)
print("Embedding length:", len(response.data[0].embedding))
