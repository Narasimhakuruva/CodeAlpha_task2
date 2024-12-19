#importing natural language toolkit
import nltk
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Downloading NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initializing stemmer algorithm, it is used to stemming the words
stemmer = PorterStemmer()

# Defining the function to preprocess the user input (if you want to use it for other purposes)
def preprocess_input(user_input):
    # Tokenizing inputs of user into words and removing punctuations
    tokens = word_tokenize(user_input)
    
    # Removes stopwords from the tokens example: 'is', 'the', 'and' etc.
    stop_words = set(stopwords.words('english'))
    
    #creating an empty list to store the filtered tokens
    filtered_tokens = [] 
    # using Loop to remove stopwords
    for word in tokens:
        # Converting the word to lowercase  and checking if it is not in stop_words
        if word.lower() not in stop_words:
            # If it's not in stop_words, add it to the filtered_tokens list
            filtered_tokens.append(word)
    
    # Applying stemming algorithm to the filtered tokens to reduce the words to their root form example: 'running' to 'run'
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

# Defining chatbot patterns and responses using NLTK chat module and reflections dictionary
pairs = [
    (r"hi|hello|hey", ["Hello! I'm your chatbot. How can I assist you? If you don't want to make conversation, type 'exit' to end."]),
    (r"how are you?", ["I'm good, thank you! How about you?", "I'm doing well, thanks for asking!"]),
    (r"what is your name?", ["I'm a chatbot created by OpenAI.", "I don't have a name, but you can call me ChatBot."]),
    (r"exit", ["Goodbye! Have a great day!", "See you later!"]),
    (r"tell me a joke", ["Why don’t skeletons fight each other? They don’t have the guts.", "Why did the math book look sad? Because it had too many problems."]),
    (r"how can you help me?", ["I can answer questions, tell jokes, provide recommendations, and much more!"]),
    (r"what is your favorite color?", ["I don't have preferences, but I think blue is cool!", "I like all colors, but blue stands out."]),
    (r"what do you do?", ["I am a chatbot here to help you with anything you need."]),
    (r"tell me a story", ["Once upon a time, there was a curious chatbot who loved helping people. It went on many adventures, answering questions and making people smile."]),
    (r"what is (.*) (your|my) (.*)", ["I am not sure about that. Can you explain it in another way?", "Could you rephrase that?"]),
    (r"(.*) (weather|forecast)", ["I can tell you about the weather! Please give me a location.", "I can help with weather updates. What city are you asking about?"]),
    (r"what is AI?", ["AI stands for Artificial Intelligence. It's a field of computer science focused on creating intelligent machines that can simulate human behavior."]),
    (r"(.*) (love|like)", ["I like helping you out with anything you need!", "I don't experience love, but I like to assist with all your questions!"]),
    (r"do you (.*)?", ["Yes, I can do a lot of things, like chatting, telling jokes, and answering questions."]),
    (r"what is (.*)", ["That's a great question! Let me think about it.", "I can try to help. Can you give me more context?"]),
    (r"who are you?", ["I'm a chatbot, created to assist you in any way I can."]),
    (r"(.*)", ["Sorry, I don't understand that. Can you rephrase?", "Could you please clarify what you mean?"])
]

# Defining function to start the chatbot conversation and take user input
def chatbot():
    print("Hello! I'm your chatbot. How can I assist you? If you don't want to make conversation, type 'exit' to end.")
    
    # Creating a Chat object with the pairs and reflections
    chat = Chat(pairs, reflections)
    
    # Starts the conversation and preprocess user input before matching patterns
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':  # Checks if the user wants to end the conversation?
            print("Goodbye! Have a great day!")
            break
        
        # You can preprocess the input for any other use or analysis, but for response matching, use the raw input
        preprocessed_input = preprocess_input(user_input)
        
        # Respond using the raw user input for pattern matching
        print("Bot: " + chat.respond(user_input))

# Runs the chatbot
if __name__ == "__main__":
    chatbot()
