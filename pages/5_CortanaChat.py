import streamlit as st
import google.genai as genai  # 👈 correct import

# 🔑 Your API Key (you can later move this to a .env file)
API_KEY = "AIzaSyBLyO2MMD7wKunNLQfBsG0ykGwHtjVq0PM"

# Create Gemini client
client = genai.Client(api_key=API_KEY)

# Page configuration
st.set_page_config(page_title="Cortana Gym Assistant", page_icon="💪", layout="centered")

st.title("💬 Chat with Cortana about Gym and Nutrition")

st.image("gif/cortana.gif")

st.write("Ask your questions about training or nutrition and **Cortana** will reply 🤖💡")

# Save chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Your question:", placeholder="Example: How much protein do I need per day?")

if st.button("Ask"):
    if user_input.strip() != "":
        try:
            # Call Gemini with Cortana style
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"You are Cortana, a helpful fitness and nutrition assistant. "
                         f"Answer clearly and concisely.\n\nUser: {user_input}"
            )

            # Save to history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Cortana", response.text))

        except Exception as e:
            st.error(f"Error connecting to Cortana (Gemini): {e}")

# Display history
st.write("### 📝 Conversation History")
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**👤 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")
