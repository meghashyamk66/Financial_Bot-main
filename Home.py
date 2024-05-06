import streamlit as st

def reformulate_question(question):
  # Replace this with your logic to reformulate the question
  # This is a simple example that just returns the question as is.
  return question

def get_chatbot_page(choice):
  if choice == "Financial Advisor (Finley)":
    # Replace this with the actual code for your Financial Advisor chatbot
    # st.write("This is the Financial Advisor chatbot page!")
    st.switch_page("pages/1_Finley.py")
  elif choice == "Investment Specialist (Lex)":
    # Replace this with the actual code for your Investment Specialist chatbot
    # st.write("This is the Investment Specialist chatbot page!")
    st.switch_page("pages/2_Lex.py")
  else:
    st.write(f"Invalid choice: {choice}")

st.title("Welcome to your Financial Assistant!")
st.write("Get personalized advice from our expert chatbots.")

# Display a brief description of each chatbot
st.markdown("**Financial Advisor (Finley):**")
st.write("Finley can help you understand your risk tolerance, set financial goals, and develop a personalized investment strategy.")

st.markdown("**Investment Specialist (Lex):**")
st.write("Lex can answer specific questions about different investment options and their potential risks and returns.")

# Create buttons for each chatbot
choice = st.selectbox("Who would you like to chat with?", ["Financial Advisor (Finley)", "Investment Specialist (Lex)"])

if st.button("Chat Now"):
  get_chatbot_page(choice)


