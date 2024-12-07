import os
import json
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# Ensure a directory exists for storing chat histories
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

class ChatManager:
    def __init__(self):
        # Initialize the language model
        self.model = Ollama(model="llama3.2")
        
        # Define the template for the chat
        template = """
        You are a helpful AI assistant. 
        Conversation History: {context}
        
        Answer the following question based on the context:
        Question: {question}
        
        Helpful Answer:"""
        
        # Create a prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"], 
            template=template
        )
        
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Create the chain
        self.chain = self.prompt | self.model | self.output_parser

    def generate_response(self, context, user_input):
        """Generate a response using the LLM chain"""
        try:
            result = self.chain.invoke({
                "context": context, 
                "question": user_input
            })
            return result
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm sorry, but I encountered an error while processing your request."

    def save_chat_history(self, chat_history):
        """Save chat history to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CHAT_HISTORY_DIR, f"chat_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(chat_history, f, indent=4)
        
        return filename

    def load_chat_histories(self):
        """Load all chat history files"""
        chat_histories = []
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(CHAT_HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    chat_histories.append({
                        'filename': filename,
                        'data': json.load(f)
                    })
        return sorted(chat_histories, key=lambda x: x['filename'], reverse=True)

def main():
    st.title("💬 Ollama Chat Assistant")
    
    # Initialize chat manager
    chat_manager = ChatManager()
    
    # Sidebar for chat history
    st.sidebar.title("Chat History")
    view_mode = st.sidebar.radio("Choose View", 
                                 ["Current Chat", "Previous Chats"])
    
    if view_mode == "Previous Chats":
        # Display previous chat histories
        histories = chat_manager.load_chat_histories()
        
        if histories:
            selected_chat = st.sidebar.selectbox(
                "Select a Chat", 
                [h['filename'] for h in histories]
            )
            
            # Find the selected chat's data
            selected_data = next(
                h['data'] for h in histories 
                if h['filename'] == selected_chat
            )
            
            st.write(f"### Chat: {selected_chat}")
            for entry in selected_data:
                if entry.get('role') == 'user':
                    st.write(f"**You:** {entry['content']}")
                elif entry.get('role') == 'assistant':
                    st.write(f"**AI:** {entry['content']}")
        else:
            st.sidebar.write("No chat histories found.")
    
    else:
        # Initialize session state for conversation
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to chat about?"):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Prepare context
            context = "\n".join([
                f"User: {msg['content']}" if msg['role'] == 'user' else f"AI: {msg['content']}" 
                for msg in st.session_state.messages
            ])
            
            # Generate response
            response = chat_manager.generate_response(context, prompt)
            
            # Add AI response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Option to save chat
            if st.button("Save Chat History"):
                saved_file = chat_manager.save_chat_history(st.session_state.messages)
                st.success(f"Chat saved to {saved_file}")

if __name__ == "__main__":
    main()