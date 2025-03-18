import ollama

# Call the Ollama model to generate a response  
response = ollama.chat(
    model="deepseek-r1:1.5b",  # Specifies the DeepSeek R1 model (1.5B parameters)
    messages=[
        {"role": "user", "content": "Explain Newton's second law of motion"},  # User's input query
    ],
)

# Print the chatbot's response
print(response["message"]["content"])  # Extracts and displays the generated response from the model