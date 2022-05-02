# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

AMUSEMENT = 'amusement'
ANGER = 'anger'
AWE = 'awe'
CONCENTRATION = 'concentration'
CONFUSION = 'confusion'
CONTEMPT = 'contempt'
CONTENTMENT = 'contentment'
DESIRE = 'desire'
DISSAPOINTMENT = 'dissapointment'
DOUBT = 'doubt'
ELATION = 'elation'
INTEREST = 'interest'
PAIN = 'pain'
SADNESS = 'sadness'
SURPRISE = 'surprise'
TRIUMPH = 'triumph'

EMOTIONS = [
    AMUSEMENT,
    ANGER,
    AWE,
    CONCENTRATION,
    CONFUSION,
    CONTEMPT,
    CONTENTMENT,
    DESIRE,
    DISSAPOINTMENT,
    DOUBT,
    ELATION,
    INTEREST,
    PAIN,
    SADNESS,
    SURPRISE,
    TRIUMPH
]

def emotion_to_int(emotion):
    for i in range(len(EMOTIONS)):
        if EMOTIONS[i] == str(emotion):
            return i
    raise Exception(f'bad emotion: {emotion}')

def emotion_from_int(i):
    try:
        value = EMOTIONS[i]
        return value
    except:
        raise Exception(f'bad emotion index: {i}')
