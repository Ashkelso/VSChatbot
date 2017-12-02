
# Author: Ashley Kelso
# this is a victims compensation chat bot designed to lead users through a series of questions to determine what
# assistance they may be entitled to and to refer them to helpful services

# Author: Victor Aung
# refactored and created as web app which can be browsed and call API services running as a server
# to get greetings and advice from the chatbot state machine and NLP engines
# runnable on a localhost and given port

import numpy as np
import pickle
import NLPClassifier
import requests
import json

from flask import Flask
from flask import request
from flask import render_template
#from urllib2 import Request, urlopen, URLError
from NLPClassifier import preprocessTextAdvanced

app = Flask(__name__)

url = "http://localhost:8770"
api_Greetings = url
api_Advice = url + '/answer'

def get_advice(user_response, state):
    payload = {'user_response': user_response, 'category': state['category'], 'previous_question': state['previous_question']}
    myResponse = requests.get(api_Advice, params=payload)
    #http://localhost:8765/answer?user_response=bye&category=&previous_question=main_menu

    # For successful API call, response code will be 200 (OK)
    if (myResponse.ok):

        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)

        state['advice'] = jData['advice']
        state['category'] = jData['category']
        state['previous_question'] = jData['previous_question']
    else:
        # If response code is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()

@app.route('/')
def say_greetings():
    # for tracking the state of the conversation and recording the category of abuse - 'assault' or 'sexual_abuse'
    state = {'category': "",
             'previous_question': "",
             'advice': ""}

#    myResponse = requests.get(url,auth=HTTPDigestAuth(raw_input("username: "), raw_input("Password: ")), verify=True)
    myResponse = requests.get(api_Greetings)

    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):

        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)

        state['advice'] = jData['advice']
        state['previous_question'] = jData['previous_question']
    else:
      # If response code is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()

    return render_template("webchatbot.html", result='\n'+state['advice'], category=state['category'], previous_question=state['previous_question'])

@app.route('/answer', methods=['GET'])
def take_response():
    # for tracking the state of the conversation and recording the category of abuse - 'assault' or 'sexual_abuse'
    state = {'category': "",
             'previous_question': "",
             'advice': ""}

    user_response = request.args.get('user_response')
    state['advice'] = request.args.get('advice') #recyle previous advice if null response is submitted
    state['category'] = request.args.get('category')
    state['previous_question'] = request.args.get('previous_question')
    if (user_response != ''):
        state['advice'] = ''
        get_advice(user_response=user_response, state=state)
        if (user_response == 'bye'):
            return render_template("webchatbot_bye.html", result='\n'+state['advice'])
        else:
            return render_template("webchatbot.html", result='\n' + state['advice'], category=state['category'],
                                   previous_question=state['previous_question'])
    else:
        # recycle the page
        return render_template("webchatbot.html", result=state['advice'], category=state['category'],
                               previous_question=state['previous_question'])


if __name__ == "__main__":
    app.run(host='localhost', port=8760)
