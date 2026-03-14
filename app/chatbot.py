# import random
# import json
# from app.config import INTENTS_PATH
# from app.inference import predict_class

# with open(INTENTS_PATH) as f:
#     intents = json.load(f)

# def get_response(user_input):
#     results = predict_class(user_input)
#     if not results:
#         return "I'm sorry, I don't quite understand that."
    
#     tag = results[0]['intent']
#     list_of_intents = intents['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "I'm not sure how to help with that."





import random
import json
from app.config import INTENTS_PATH
from app.inference import predict_class

with open(INTENTS_PATH) as f:
    intents = json.load(f)

def get_response(user_input):
    results = predict_class(user_input)
    
    # FALLBACK LOGIC: If results is empty, the model isn't confident enough
    if not results:
        return (
            "I'm sorry, I'm specifically trained to talk about Yusuf's portfolio, "
            "his AI projects, and movie recommendations. I didn't quite get that—"
            "could you try asking about his location, skills, or social media?"
        )

    # If we have a result, proceed as normal
    tag = results[0]['intent']
    # Professional touch: get the probability to show how sure the bot is
    prob = float(results[0]['probability'])
    
    for i in intents['intents']:
        if i['tag'] == tag:
            # You can even print this to your terminal for debugging
            print(f"[DEBUG] Predicted: {tag} | Confidence: {prob*100:.2f}%")
            return random.choice(i['responses'])
            
    return "I encountered a technical glitch. Please try again!"