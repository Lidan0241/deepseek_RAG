import gradio as gr
# Define a Gradio interface
interface = gr.Interface(
    fn=ask_question,  # The function that processes user input and generates a response (logic of the app)
    inputs=[
        gr.File(label="Upload PDF (optional)"),  # Optional file upload input for a PDF document
        gr.Textbox(label="Ask a question")  # Text input where the user types their question
    ],
    outputs="text",  # The function returns a text response
    title="Ask questions about your PDF",  # The title displayed on the interface
    description="Use DeepSeek-R1 1.5B to answer your questions about the uploaded PDF document.",  # Brief description of the interface's functionality
)

# Launch the Gradio interface to start the web-based app
interface.launch()