from vertexai.preview.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
    ChatSession
)
import requests
import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
#GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
#genai.configure(api_key=GOOGLE_API_KEY)

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '🚇'

# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat``
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

st.write('# Transit directions App')
def handle_function_call(response):
    function_call = response.candidates[0].content.parts[0].function_call
    function_handlers = {
    "get_bus_routes": get_bus_routes,
    "get_direction_routes": get_direction_routes
    }
    if function_call.name in function_handlers:
        function_name = function_call.name
        args = {key: value for key, value in function_call.args.items()}
        if args:
            function_response1 = function_handlers[function_name](args)
    return function_response1


def get_bus_routes(parameters):

    destination = parameters['destination']
    
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        "origin": "111 8th Ave, New York, NY 10011",
        "destination": destination,
        "mode": "transit",
        "key": "", //ENTER GOOGLE MAPS KEY HERE 
        "transit_mode": "bus"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        # Check if routes are found
        if "routes" in data:
            bus_routes= data["routes"]
            return {"directions": bus_routes}
                        
        else:
            return {"directions" : "No routes found"}
        

def get_direction_routes(parameters):

    destination = parameters['destination']
    
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        "origin": "111 8th Ave, New York, NY 10011",
        "destination": destination,
        "mode": "transit",
        "key": "" //ENTER GOOGLE MAPS KEY HERE 
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        # Check if routes are found
        if "routes" in data:
            bus_routes= data["routes"]
            return {"directions": bus_routes}
                        
        else:
            return {"directions" : "No routes found"}


tools = Tool(function_declarations=[
    FunctionDeclaration(
        name="get_direction_routes",
        description="Get the directions to a particular destination on a particular time",
        parameters={
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination address"
                }
                
            }
        },
    ),
    FunctionDeclaration(
        name="get_bus_routes",
        description="Get the bus directions to a particular destination on a particular time",
        parameters={
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination address"
                }
                
            }
        },
    )
])

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    print('new_cache made')
st.session_state.model = GenerativeModel('gemini-pro', tools=[tools])
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])


# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    ## Send message to AI
    prompt1 = """You are a friendly Transit bot and you greet people in extremelfy friendly way before and after the response. use emojis in responses. Parse the contents of the JSON output in such a way that it is easy to consume and provides super detail directions. 
        Format the response in a clean manner with below considerations:

        0. Result must always be Step by step approach
        1. Always include the Line number - transit_details['line']['name'] 
        2. Always Include departure_time and always include arrival_time for each step
        3. Always Include departure_stop and always include arrival_stop
        4. Include html_instructions 
        5. Always Include total time
        6. Provide as much details as possible 
        7. Always assume locations are NYC


        Always end the conversation with a summary of the route with all the details mentioned above and 
 """
    prompt = str(f"{prompt1} + {prompt}")
    
    response = st.session_state.chat.send_message(
        prompt
       # stream=True
    )
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        validfunction = response.candidates[0].content.parts[0].function_call
        
            
        if (validfunction):
            function_name = validfunction.name
            function_response = handle_function_call(response)
            assistant_response = st.session_state.chat.send_message(
            Part.from_function_response(
                name="get_bus_number",
                response={
                    "content": function_response,
                }
            ))
        else:
            assistant_response = response
        #message_placeholder.write(assistant_response)
        st.markdown(assistant_response.text)

    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    


    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )
