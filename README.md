# Chat_G
# Installation and Setup Instructions
1. Install Dependencies:
Use the provided pip command to install all necessary libraries:
```
pip install Flask Flask-SQLAlchemy pyngrok transformers pillow fitz pdfplumber python-pptx torch torchvision groq-client python-docx
```
install pytesseract if you have Tesseract on your system as it requires to be install manually on the system else we still have tried to get the accurate summary of the document.
2. Set Up the Flask App:
Save the provided code into a Python file, e.g., `app.py`.
3. Run the Flask App:
Execute the Flask application using the following command:
```
python app.py
```
4. Access the App:
Use the URL provided by ngrok (displayed in the console) to access the application in your browser.
5. Upload Documents:
Upload PDF, PPT, or Word documents to extract and summarize their content. Ask questions based on the summaries.
