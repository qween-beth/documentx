import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64

def allowed_file(filename):
    """Check if file has an allowed image extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_with_gemini(image, api_key):
    """Extract text from image using Google Gemini Vision API."""
    try:
        # Configure Gemini API with the provided API key
        genai.configure(api_key=api_key)
        
        # Convert the PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Base64 encode the image content
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        
        # Prepare the image input
        image_data = {
            "mime_type": "image/png",
            "data": encoded_image,
        }
        
        # Prompt for the model
        prompt = """Analyze the text in the provided image. Extract all readable content 
                   and present it in a structured Markdown format that is clear, concise, 
                   and well-organized."""
        
        # Generate response using Gemini
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([image_data, prompt])
        
        if not response or not hasattr(response, "text"):
            raise ValueError("Invalid response from Gemini API")
        
        return response.text
    except Exception as e:
        st.error(f"Gemini text extraction error: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Image Text Extraction", layout="centered")
    
    # Title and description
    st.title("Image Text Extraction")
    st.markdown("Upload an image to extract text using Google's Gemini Vision API")
    
    # API key input
    st.subheader("Enter your API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your API key here..."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']
    )
    
    # Proceed only if both API key and file are provided
    if api_key and uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Extract text button
            if st.button("Extract Text"):
                with st.spinner("Extracting text..."):
                    try:
                        extracted_text = extract_text_with_gemini(image, api_key)
                        
                        # Display extracted text
                        st.subheader("Extracted Text")
                        st.markdown(extracted_text)
                        
                        # Add download button for extracted text
                        st.download_button(
                            label="Download Extracted Text",
                            data=extracted_text,
                            file_name="extracted_text.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error extracting text: {str(e)}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    elif not api_key:
        st.warning("Please enter your API key to proceed.")
    elif uploaded_file is None:
        st.warning("Please upload an image file to proceed.")

if __name__ == "__main__":
    main()
