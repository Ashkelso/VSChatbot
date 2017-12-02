
# Author: Ashley Kelso
# this is a victims compensation chat bot designed to lead users through a series of questions to determine what
# assistance they may be entitled to and to refer them to helpful services

# Author: Victor Aung
# refactored and created RESTful API classes which can be called from an external UI e.g. web app, FB or slack
# to get greetings and advice on demand from this chatbot state machine and NLP classifiers
# testable as a localhost on a given port

import numpy as np
import pickle
import NLPClassifier

from flask import Flask
from flask import request
from flask import render_template
from flask_restful import Resource, Api

from NLPClassifier import preprocessTextAdvanced

app = Flask(__name__)
api = Api(app)

# Loading classifiers
forest = pickle.load(open("RandomForest.p","rb"))
supportV = pickle.load(open("NuSVC.p","rb"))
bayes = pickle.load(open("MultiNB.p","rb"))

def response(key):
    #dictionary of all the questions and responses for the chatbot
    conversations = {'welcome' :"Hi, I'm Vic, a chatbot. \nMy job is to help people with information about "
                                "victims support services in New South Wales. If you have been the "
                                "victim of physical or sexual abuse I'd like to help you \n\n",

                     'main menu': "What would you like to talk about?\n"
                                  "* Free legal services \n"
                                  "* Counselling \n"
                                  "* Victims Compensation \n\n"
                                  "When you're done talking to me, just say 'bye'\n\n",

                     'limit': "That's everything I know about. \n\n",

                     'bye': "Thanks for chatting with me, I hope i was able to be helpful.\n I hope "
                            "you have a good day and that things take a turn for the better :)\n\n",

                     'error': "Sorry, I didn't understand that. I'm still learning. "
                              "Thank you for being patient with me. Could you rephrase your response?\n\n",

                     'victims compensation': "Victims Services is a government-funded agency that "
                                             "provides assistance to people who have been injuried "
                                             "by the violent conduct of others \n\n"
                                             "This assistance includes: \n"
                                             "1. A lump sum payment (called a 'recognition payment) to "
                                             "acknowledge the trauma that you experienced \n"
                                             "2. Free counselling with experienced social workers and "
                                             "psychologists. \n"
                                             "3. Financial assistance for lost/damaged property, lost wages "
                                             "and urgent needs related to the violence you suffered \n\n",

                     'briefly': "I'll need to ask you some questions to determine what assistance "
                                "you may be eligible for. \n\n"
                                "Could you briefly describe what the offender did to you?\n\n",

                     'sexual abuse': "It sounds like you experienced a form of sexual abuse. Sexual abuse "
                                     "includes actions like inappropriate touching ('indecent assault'), sexual assault"
                                     "(penetration without consent),and attempted sexual assault."
                                     "\n\nWould you agree that any of these descriptions match what happened "
                                     "to you? \n\n",

                     'further questions': "I understand that this is difficult to talk about, but I'll need some"
                                          " further details about what happened \n\n",

                     'match error': "Sorry, i'm still learning. Would you like me to give you the contact details for "
                                    "some free legal services who can help you with advice and applying for some "
                                    "assistance?\n\n",

                     'sexual series': "Did you experience physical or sexual abuse from the same offender "
                                      "on any other occasions?\n\n",

                     'aggravated': "When you suffered this abuse, did the offender use a weapon or have the help "
                                   "of another person?\n\n",

                     'penetrate': "Just to confirm, did the offender penetrate any part of your body?\n\n",

                     'attempt penetrate': "Did the offender attempt to penetrate any part of your body?\n\n",

                     'injuries': "Did you suffer serious injury as a result of the abuse, e.g. did you need to be "
                                 "admitted to hospital for the treatment of those injuries (Yes/No)? \n\n",

                     'Category B': "Based on your answers you should be entitled to: \n"
                                   "1. A Category B recognition payment of $10,000 \n"
                                   "2. 10 free counselling sessions \n"
                                   "3. Financial assistance for lost wages if you needed time off work, or if any of "
                                   "your property was damaged or stolen during the attack \n\n",

                     'Category C': "Based on your answers you should be entitled to: \n"
                                   "1. A Category C recognition payment of $5,000 \n"
                                   "2. 10 free counselling sessions \n"
                                   "3. Financial assistance for lost wages if you needed time off work, or if any of"
                                   " your property was damaged or stolen during the attack \n\n",

                     'Category D': "Based on you answers you should be entitled to: \n"
                                   "1. A Category D recognition payment of $1,500 \n"
                                   "2. 10 free counselling sessions \n"
                                   "3. Financial assistance for lost wages if you needed time off work, or if any of your"
                                   "property was damages or stolen during the attack \n\n",

                     'assault': "It sounds like you suffered an assault. An assault can be where someone hits you, or"
                                " where they cause you to fear that they are about to do you physical harm. Does this "
                                "sound like what you experienced? \n\n",

                     'serious harm': "Did you suffer serious injuries as a result of the attack? "
                                     "e.g. Did you need to be admitted to hospital for the treatment of your injuries"
                                     " (Yes/No)? \n\n",

                     'eighteen':"Were you under the age of 18 when you were assaulted? \n\n",

                     'child': "Were you assaulted by the same offender on more than one occasion while you were under the "
                              "age of 18 years old? \n\n",

                     'application': "To make your application for for this support you'll  need: \n\n"
                                    "1. Proof of what the offender did to you. \n\n"
                                    "If at all possible, make a report to police about what happened and get a copy to "
                                    "include with your application. Otherwise, you can write out what happened in a "
                                    "statutory declaration (you can get the statutory declaration from from a news agent, "
                                    "or you can download the form from here: http://ow.ly/bbnG30fnzT1) \n\n"
                                    "2. Proof of the harm you suffered from the offender's actions.\n\n"
                                    "This can be a letter from your GP setting out what happened and the injuries you "
                                    "suffered. Or you can obtain a copy of your hospital notes, if you attended a hospital "
                                    "for your injuries. If your injuries are psychological then you can obtain a short "
                                    "report from your psychologist setting out what happened and the impact it has had on "
                                    "your mental health. \n\n"
                                    "You can contact Victims Services to start your application (Ph: 1800 633 063 / Email: "
                                    "vs@justice.nsw.gov.au). \n\n",

                     'application advice': "Would you like information on free legal services that can assist you with making "
                                           "your application for Victims Support? \n\n",

                     'disclaimer': "Remember that i'm just a chatbot, not a lawyer, can only"
                                   " provide information, not legal advice. \n\n",

                     'counselling': "Two options for free counselling include: \n"
                                    "1. See your GP for a mental health plan and referral for Medicare-funded "
                                    "counselling with a psychologist\n"
                                    "2. Apply to Victims Services for NSW government-funded counselling with "
                                    "social workers and psychologists (Ph: 1800 633 063 / Email: vs@justice.nsw.gov.au)\n\n",

                     'other services': "Would you like to find out what other assistance may be available to you through "
                                       "Victims Services (such as financial assistance and compensation)?\n\n",

                     'legal services': "It is important to know your rights if you've been a victim of crime."
                                       "Would you like some information on free legal services? \n\n",

                     'free legal services': "Here's some information on free legal services: \n\n"
                                            "Women's Legal Service NSW:\n"
                                            "Website: http://www.wlsnsw.org.au/ \n"
                                            "Phone: 02 8745 6900 \n"
                                            "Email: reception@wlsnsw.org.au \n\n"
                                            "Domestic Violence Legal Service: \n"
                                            "Phone: 1800 810 784 \n\n"
                                            "Indigenous Women’s Legal Program: \n"
                                            "Phone: 1800 639 784 \n\n"
                                            "Community Legal Centres can also help with free advice and referrals. \n"
                                            "here's the link to their contact details: http://www.clcnsw.org.au/clc_"
                                            "directory.php \n\n",

                     'sexual abuse information': "Here's a link to some information on your legal rights: \n"
                                                 "http://www.wlsnsw.org.au/resources/sexual-assault/ \n\n" ,

                     'AVO': "Here's some information from the Southern Women's Group about how to make an Apprehended "
                            "Violence Order work for you: http://www.sl.nsw.gov.au/sites/default/files/just_piece_paper"
                            ".pdf \n\n",

                     'attempt': "Sorry about that, could we try that again? I'm doing my best, but I'm still learning."
                                " would you mind rephrasing your response for me (yes/no)? \n\n"
                     }
    return conversations[key]

def stateManager(user_response, state):

    #for classifying simple user responses
    yes = ['yep', 'yeah', 'yes', 'yup', 'correct', "ok"]
    no = ['no', 'nah', 'nope', "didn't", "wasn't", 'not']
    legal = ["lawyer", "solicitor", "legal"]
    victimsComp = ["victims", "compensation", "victim's", "comp", "money", "financial", "victim"]
    counselling = ["counselling", "counseling", "psychologist", "therapy", "counsellor", "therapist"]

    if state['previous_question']=='main menu':
        for s in user_response.split():
            if s in victimsComp:
                state['advice'] += response('victims compensation')
                state['advice'] += response('briefly')
                state['previous_question'] = 'briefly'
                break
            elif s in counselling:
                state['advice'] += response('counselling')
                state['advice'] += response('other services')
                state['previous_question'] = 'other services'
                break
            elif s in legal:
                state['advice'] += response('free legal services')
                state['advice'] += response('main menu')
                state['previous_question'] = 'main menu'
                break
            elif s in no:
                state['advice'] += response('limit')
                state['advice'] += response('bye')
                state['previous_question'] = 'bye'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response('main menu')
                    state['previous_question'] = 'main menu'

    elif state['previous_question'] == 'other services':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('victims compensation')
                state['advice'] += response('briefly')
                state['previous_question'] = 'briefly'
                break
            elif s in no:
                state['advice'] += response('legal services')
                state['previous_question'] = 'legal services'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'legal services':
        state['advice'] += response('free legal services')
        state['advice'] += response('main menu')
        state['previous_question'] = 'main menu'

    elif state['previous_question'] == 'briefly':
        category = classifier(user_response)
        state['advice'] += response(category)
        state['previous_question'] = category

#sexual abuse questions
    elif state['previous_question'] == 'sexual abuse':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('further questions')
                state['category'] = 'sexual abuse'
                state['advice'] += response('sexual series')
                state['previous_question'] = 'sexual series'
                break
            elif s in no:
                state['advice'] += response('attempt')
                state['previous_question'] = 'attempt'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'sexual series':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category B')
                state['previous_question'] = 'Category B'
                application(state)
                break
            elif s in no:
                state['advice'] += response('aggravated')
                state['previous_question'] = 'aggravated'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'aggravated':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category B')
                state['previous_question'] = 'Category B'
                application(state)
                break
            elif s in no:
                state['advice'] += response('penetrate')
                state['previous_question'] = 'penetrate'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'penetrate':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category C')
                state['previous_question'] = 'Category C'
                application(state)
                break
            elif s in no:
                state['advice'] += response('attempt penetrate')
                state['previous_question'] = 'attempt penetrate'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'attempt penetrate':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('injuries')
                state['previous_question'] = 'injuries'
                break
            elif s in no:
                state['advice'] += response('Category D')
                state['previous_question'] = 'Category D'
                application(state)
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'injuries':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category C')
                state['previous_question'] = 'Category C'
                application(state)
                break
            elif s in no:
                state['advice'] += response('Category D')
                state['previous_question'] = 'Category D'
                application(state)
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

#handleing re-attempts at classifying the offence
    elif state['previous_question']=='attempt':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('briefly')
                state['previous_question'] = 'briefly'
                break
            elif s in no:
                state['advice'] += response('match error')
                state['previous_question'] = 'match error'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'match error':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('free legal services')
                state['advice'] += response('main menu')
                state['previous_question'] = 'main menu'
                break
            elif s in no:
                state['advice'] += response('main menu')
                state['previous_question'] = 'main menu'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

#assault questions
    elif state['previous_question'] == 'assault':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('further questions')
                state['category'] = 'assault'
                state['advice'] += response('serious harm')
                state['previous_question'] = 'serious harm'
                break
            elif s in no:
                state['advice'] += response('attempt')
                state['previous_question'] = 'attempt'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'serious harm':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category C')
                state['previous_question'] = 'Category C'
                application(state)
                break
            elif s in no:
                state['advice'] += response('eighteen')
                state['previous_question'] = 'eighteen'
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'eighteen':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('child')
                state['previous_question'] = 'child'
                break
            elif s in no:
                state['advice'] += response('Category D')
                state['previous_question'] = 'Category D'
                application(state)
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'child':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('Category C')
                state['previous_question'] = 'Category C'
                application(state)
                break
            elif s in no:
                state['advice'] += response('Category D')
                state['previous_question'] = 'Category D'
                application(state)
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')

    elif state['previous_question'] == 'application':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('legal services')
                state['previous_question'] = 'legal services'
                break
            elif s in no:
                state['advice'] += response('bye')
                break
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    elif state['previous_question'] == 'application advice':
        for s in user_response.split():
            if s in yes:
                state['advice'] += response('free legal services')
                state['advice'] += response('main menu')
                state['previous_question'] = 'main menu'
            elif s in no:
                state['advice'] += response('main menu')
                state['previous_question'] = 'main menu'
            else:
                if s == user_response.split()[-1]:
                    state['advice'] += response('error')
                    state['advice'] += response(state['previous_question'])

    else:
        state['advice'] += response('error')
        state['advice'] += response(state['previous_question'])

def application(state):
    state['advice'] += response('application')
    # providing further resources based on category of offence
    state['advice'] += '\n\n you may also find this information helpful:\n\n'
    if state['category'] == 'sexual abuse':
        state['advice'] += response('sexual abuse information')
    else:
        state['advice'] += response('AVO')
    state['advice'] += response('disclaimer')
    state['advice'] += response('application advice')
    state['previous_question'] = 'application advice'

def classifier(user_response):
    #call trained classifier to discern 'assault' or 'sexual abuse'

    vote = {'assault': 0,
            'sexual abuse': 0}

    s = supportV.predict(np.array([user_response]))[0]
    print ('s=',s)

    vote[s] += 1

    r = forest.predict(np.array([user_response]))[0]
    print ('r=',r)

    vote[r] += 1

    b = bayes.predict(np.array([user_response]))[0]
    print ('b=',b)

    vote[b] += 1

#    if vote['assault'] > vote['sexual abuse']:
    if vote['assault'] > vote['sexual abuse']:
            print('score SA = ', vote['sexual abuse'], 'score assault = ', vote['assault'])
            return 'assault'
    else:
        print('score SA = ', vote['sexual abuse'], 'score assault = ', vote['assault'])
        return 'sexual abuse'


    #return supportV.predict(np.array([user_response]))[0])

    # for s in user_response.split():
    #     if(s in ['vagina','forced','penetrated','penis','raped','sexually','touched','breast','breasts','dick','raping']):
    #         return 'sexual abuse'
    #     elif(s in ['hit', 'hits','punched','bashed','beat','beats','domestic','fist','assault','assaulted']):
    #         return 'assault'
    #     else:
    #         if s == user_response.split()[-1]:
    #             return 'error'

class Greetings(Resource):
    def get(self):
        # for tracking the state of the conversation and recording the category of abuse - 'assault' or 'sexual abuse'
        state = {'category': "",
                 'previous_question': "",
                 'advice': ""}

        state['advice'] = response('welcome')
        state['advice'] += response('main menu')

        # setting the initial state of the conversation
        state['previous_question'] = 'main menu'

        return {'advice': '\n' + state['advice'],
                'category':state['category'],
                'previous_question': state['previous_question']}

class Bye(Resource):
    def get(self):
        # for tracking the state of the conversation and recording the category of abuse - 'assault' or 'sexual abuse'
        state = {'category': "",
                 'previous_question': "",
                 'advice': ""}

        state['advice'] = response('bye')

        return {'advice': '\n' + state['advice'],
                'category':state['category'],
                'previous_question': state['previous_question']}

class Advice(Resource):
    def get(self):
        # for tracking the state of the conversation and recording the category of abuse - 'assault' or 'sexual abuse'
        state = {'category': "",
                 'previous_question': "",
                 'advice': ""}

        user_response = request.args.get('user_response')
        state['category'] = request.args.get('category')
        state['previous_question'] = request.args.get('previous_question')
        if (user_response != ''):
            state['advice'] = ''
            if (user_response == "bye"):
                state['advice'] = response('bye')
            else:
                stateManager(user_response=user_response, state=state)
            return {'advice': '\n' + state['advice'],
                    'category': state['category'],
                    'previous_question': state['previous_question']}

api.add_resource(Greetings, '/')
api.add_resource(Advice, '/answer')
api.add_resource(Bye, '/bye')

if __name__ == "__main__":
    app.run(host='localhost', port=8770)
