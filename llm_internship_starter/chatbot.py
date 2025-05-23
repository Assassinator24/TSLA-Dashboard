import google.generativeai as genai

def gemini_bot(query, df):
    # Set API key directly - replace with your actual API key
    api_key = "AIzaSyA891YrLApwu1Q4PGLEdgVqLflYq3A3DxU"
    
    genai.configure(api_key=api_key)
    
    try:
        # Specify a newer Gemini model directly instead of searching
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare context and prompt
        context = df.head(50).to_csv(index=False)
        prompt = f"""You are an AI assistant analyzing TSLA stock OHLCV data.
Here is a sample of the dataset:
{context}

Answer the following question about the dataset:
{query}"""
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # If the specified model fails, try to find any available Gemini model
        try:
            available_models = [model.name for model in genai.list_models()]
            print(f"Available models: {available_models}")
            
            # Try to find any available Gemini model
            gemini_models = [model for model in available_models if "gemini" in model.lower()]
            
            if gemini_models:
                print(f"Found Gemini models: {gemini_models}")
                # Try with the first available Gemini model
                model = genai.GenerativeModel(gemini_models[0])
                
                # Prepare context and prompt
                context = df.head(50).to_csv(index=False)
                prompt = f"""You are an AI assistant analyzing TSLA stock OHLCV data.
Here is a sample of the dataset:
{context}

Answer the following question about the dataset:
{query}"""
                
                # Generate response
                response = model.generate_content(prompt)
                return response.text
            else:
                return f"No Gemini models available. Original error: {str(e)}"
        except Exception as nested_e:
            return f"Error generating response: {str(e)}\nFallback attempt failed: {str(nested_e)}\n\nPlease check your API key and ensure you have access to Gemini models."