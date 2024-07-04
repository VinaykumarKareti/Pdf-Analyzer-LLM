# PDF Analyzer LLM

**Name**: K.V. Vinay Kumar

This work is assigned as an assignment by ALEMENO.
User Interface look like:
![image](https://github.com/VinaykumarKareti/Pdf-Analyzer-LLM/assets/105053576/d51f20a9-e8e0-490d-9da5-f497cb8078a9)

=============================================================================================================================================================

This Streamlit application allows users to interact with PDF documents through a conversational AI interface. Users can upload PDF files, which are processed to extract and chunk text data. This data is then indexed in a vector store to enable efficient similarity searches. When users ask questions about the uploaded PDFs, the application uses Google Generative AI models to generate detailed answers based on the context extracted from the documents.

### File Structure

- `main.py`: Contains the primary Streamlit application code.
- `requirements.txt`: Lists all the necessary Python packages for the application.

### Key Components

1. **PDF Processing**:
    - Extracts text from uploaded PDF files using `PyPDF2`.
    - Splits the extracted text into manageable chunks using `RecursiveCharacterTextSplitter`.

2. **Vector Store Creation**:
    - Embeds the text chunks using the Google Generative AI Embeddings model.
    - Stores the embeddings in a FAISS vector store (Storing vectors locally) for efficient similarity searches.

3. **Conversational Chain**:
    - Constructs a prompt template to generate detailed answers using the Gemini model (`google/gemma-2b-it`).
    - Generates responses to user questions based on the context retrieved from the vector store.

4. **User Interaction**:
    - Provides an interface for users to upload PDF files.
    - Allows users to input questions and receive answers based on the uploaded PDFs.
    - Displays the conversation history.

### How to Run the Application

#### Prerequisites

1. Python 3.7 or higher
2. All necessary packages listed in `requirements.txt`

#### Setup

1. **Clone the Repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and Activate Virtual Environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:
    - Create a `.env` file in the root directory.
    - Add your Google API key: (API key for using google Embedding model)
        ```env
        GOOGLE_API_KEY=your_api_key_here
        ```

#### Running the Application

1. **Start the Streamlit Application**:
    ```sh
    streamlit run main.py
    ```

2. **Use the Application**:
    - Open the provided URL in your browser (usually `http://localhost:8501`).
    - Upload PDF files via the sidebar.
    - Input questions and view responses in the main panel.

